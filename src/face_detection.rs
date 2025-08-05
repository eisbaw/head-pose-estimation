use crate::Result;
use ndarray::{s, Array1, Array2, Array3, Array4, CowArray};
use opencv::core::{Mat, Point2f, Rect, Scalar, Size, CV_32F};
use opencv::imgproc::{self, InterpolationFlags};
use opencv::prelude::*;
use ort::{Environment, Session, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Bounding box of the detected face
    pub bbox: Rect,
    /// Confidence score of the detection
    pub score: f32,
    /// Optional keypoints (5 facial landmarks)
    pub keypoints: Option<Vec<Point2f>>,
}

/// SCRFD Face Detector using ONNX Runtime
pub struct FaceDetector {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
    input_size: (i32, i32),
    conf_threshold: f32,
    nms_threshold: f32,
    with_kps: bool,
    num_anchors: usize,
    strides: Vec<i32>,
    offset: usize,
    center_cache: HashMap<(i32, i32, i32), Array2<f32>>,
}

impl FaceDetector {
    /// Create a new face detector from an ONNX model file
    #[allow(clippy::too_many_lines)]
    pub fn new<P: AsRef<Path>>(model_path: P, conf_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("face_detector")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        let session = ort::SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        // Get model configurations from the model
        let input_meta = session.inputs.first()
            .ok_or_else(|| crate::error::Error::ModelError("Model has no inputs".to_string()))?;
        
        let input_name = input_meta.name.clone();
        let input_shape = &input_meta.dimensions;
        
        // Extract input size from shape [batch, channels, height, width]
        let input_size = if input_shape.len() >= 4 {
            // Note: dimensions are Option<i64>, we need to handle this properly
            let height = input_shape[2].unwrap_or(640) as i32;
            let width = input_shape[3].unwrap_or(640) as i32;
            (width, height)
        } else {
            (640, 640) // Default SCRFD input size
        };

        // Get output names
        let output_names: Vec<String> = session.outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        
        let num_outputs = output_names.len();
        
        // Determine model configuration based on number of outputs
        let (offset, strides, num_anchors, with_kps) = match num_outputs {
            6 => (3, vec![8, 16, 32], 2, false),
            9 => (3, vec![8, 16, 32], 2, true),
            10 => (5, vec![8, 16, 32, 64, 128], 1, false),
            15 => (5, vec![8, 16, 32, 64, 128], 1, true),
            _ => {
                log::warn!("Unknown model configuration with {} outputs, using defaults", num_outputs);
                (3, vec![8, 16, 32], 2, false)
            }
        };

        Ok(Self {
            session,
            input_name,
            output_names,
            input_size,
            conf_threshold,
            nms_threshold,
            with_kps,
            num_anchors,
            strides,
            offset,
            center_cache: HashMap::new(),
        })
    }

    /// Detect faces in an image
    pub fn detect(&mut self, image: &Mat) -> Result<Vec<FaceDetection>> {
        // Get image dimensions
        let img_height = image.rows();
        let img_width = image.cols();
        
        // Calculate scale and resize dimensions
        let ratio_img = img_height as f32 / img_width as f32;
        let (input_width, input_height) = self.input_size;
        let ratio_model = input_height as f32 / input_width as f32;
        
        let (new_width, new_height) = if ratio_img > ratio_model {
            let new_height = input_height;
            let new_width = (new_height as f32 / ratio_img) as i32;
            (new_width, new_height)
        } else {
            let new_width = input_width;
            let new_height = (new_width as f32 * ratio_img) as i32;
            (new_width, new_height)
        };
        
        let det_scale = new_height as f32 / img_height as f32;
        
        // Resize image
        let mut resized = Mat::default();
        imgproc::resize(
            image,
            &mut resized,
            Size::new(new_width, new_height),
            0.0,
            0.0,
            InterpolationFlags::INTER_LINEAR as i32,
        )?;
        
        // Create padded image
        let mut det_img = Mat::new_rows_cols_with_default(
            input_height,
            input_width,
            opencv::core::CV_8UC3,
            Scalar::all(0.0),
        )?;
        
        let mut roi = det_img.roi_mut(Rect::new(0, 0, new_width, new_height))?;
        resized.copy_to(&mut roi)?;
        
        // Preprocess image
        let inputs = self.preprocess(&det_img)?;
        
        // Run inference
        let (scores_list, bboxes_list, kpss_list) = self.forward(inputs, self.conf_threshold)?;
        
        // Postprocess results
        let detections = self.postprocess(
            scores_list,
            bboxes_list,
            kpss_list,
            det_scale,
            (img_width, img_height),
        )?;
        
        Ok(detections)
    }
    
    /// Preprocess image for ONNX model
    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>> {
        // Convert BGR to RGB and normalize
        let mut rgb_image = Mat::default();
        imgproc::cvt_color(image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
        
        // Convert to f32
        let mut float_image = Mat::default();
        rgb_image.convert_to(&mut float_image, CV_32F, 1.0, 0.0)?;
        
        // Extract data and normalize
        let height = float_image.rows() as usize;
        let width = float_image.cols() as usize;
        let channels = 3;
        
        let mut data = vec![0.0f32; height * width * channels];
        
        // Safely extract data from Mat
        for row in 0..height {
            for col in 0..width {
                for ch in 0..channels {
                    let idx = (row * width + col) * channels + ch;
                    // Access pixel value safely
                    // For 3-channel image, access as Vec3f
                    let pixel = float_image.at_2d::<opencv::core::Vec3f>(row as i32, col as i32)?[ch];
                    data[idx] = (pixel - 127.5) / 128.0;
                }
            }
        }
        
        // Reshape to NCHW format
        let mut array = Array4::from_shape_vec((1, height, width, channels), data)
            .map_err(|e| crate::error::Error::ModelError(format!("Failed to create array: {}", e)))?;
        
        // Transpose from NHWC to NCHW
        array = array.permuted_axes([0, 3, 1, 2]);
        
        Ok(array)
    }
    
    /// Run forward pass through the model
    fn forward(
        &mut self,
        inputs: Array4<f32>,
        threshold: f32,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>)> {
        let mut scores_list = Vec::new();
        let mut bboxes_list = Vec::new();
        let mut kpss_list = Vec::new();
        
        // Get input dimensions before moving the array
        let input_height = inputs.shape()[2] as i32;
        let input_width = inputs.shape()[3] as i32;
        
        // Create ONNX input
        let cow_array = CowArray::from(inputs.into_dyn());
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
        
        // Run inference
        let outputs = self.session.run(vec![input_tensor])?;
        
        // Process outputs for each stride
        for (idx, &stride) in self.strides.iter().enumerate() {
            // Extract scores
            let scores_output = outputs[idx].try_extract::<f32>()?;
            let scores_view = scores_output.view();
            let scores_flat = scores_view.as_slice().unwrap();
            let scores = Array1::from(scores_flat.to_vec());
            
            // Extract bbox predictions
            let bbox_idx = idx + self.offset;
            let bbox_output = outputs[bbox_idx].try_extract::<f32>()?;
            let bbox_view = bbox_output.view();
            let bbox_shape = bbox_view.shape();
            let n_anchors = bbox_shape[0] * bbox_shape[1] * bbox_shape[2];
            let bbox_data: Vec<f32> = bbox_view.as_slice().unwrap()
                .iter()
                .map(|&x| x * stride as f32)
                .collect();
            let bboxes = Array2::from_shape_vec((n_anchors, 4), bbox_data)
                .map_err(|e| crate::error::Error::ModelError(format!("Failed to reshape bbox: {}", e)))?;
            
            // Generate anchor centers
            let height = input_height / stride;
            let width = input_width / stride;
            let key = (height, width, stride);
            
            let anchor_centers = if let Some(centers) = self.center_cache.get(&key) {
                centers.clone()
            } else {
                let centers = self.generate_anchor_centers(height, width, stride);
                if self.center_cache.len() < 100 {
                    self.center_cache.insert(key, centers.clone());
                }
                centers
            };
            
            // Filter by threshold
            let pos_inds: Vec<usize> = scores.iter()
                .enumerate()
                .filter_map(|(i, &score)| if score >= threshold { Some(i) } else { None })
                .collect();
            
            // Convert distances to bboxes
            let decoded_bboxes = self.distance_to_bbox_array(&anchor_centers, &bboxes, None);
            
            // Collect positive detections
            let pos_scores = Array1::from(
                pos_inds.iter().map(|&i| scores[i]).collect::<Vec<f32>>()
            );
            let pos_bboxes = Array2::from_shape_vec(
                (pos_inds.len(), 4),
                pos_inds.iter()
                    .flat_map(|&i| decoded_bboxes.row(i).to_vec())
                    .collect(),
            ).map_err(|e| crate::error::Error::ModelError(format!("Failed to collect bboxes: {}", e)))?;
            
            scores_list.push(pos_scores);
            bboxes_list.push(pos_bboxes);
            
            // Process keypoints if available
            if self.with_kps {
                let kps_idx = idx + self.offset * 2;
                let kps_output = outputs[kps_idx].try_extract::<f32>()?;
                let kps_view = kps_output.view();
                let kps_data: Vec<f32> = kps_view.as_slice().unwrap()
                    .iter()
                    .map(|&x| x * stride as f32)
                    .collect();
                
                let kpss = self.distance_to_kps_array(&anchor_centers, &kps_data, None)?;
                let pos_kpss = Array3::from_shape_vec(
                    (pos_inds.len(), 5, 2),
                    pos_inds.iter()
                        .flat_map(|&i| kpss.slice(s![i, .., ..]).iter().cloned().collect::<Vec<f32>>())
                        .collect(),
                ).map_err(|e| crate::error::Error::ModelError(format!("Failed to collect kpss: {}", e)))?;
                
                kpss_list.push(pos_kpss);
            }
        }
        
        Ok((scores_list, bboxes_list, kpss_list))
    }
    
    /// Generate anchor centers for a given stride
    fn generate_anchor_centers(&self, height: i32, width: i32, stride: i32) -> Array2<f32> {
        let mut centers = Vec::new();
        
        for y in 0..height {
            for x in 0..width {
                let cx = (x * stride) as f32;
                let cy = (y * stride) as f32;
                
                if self.num_anchors > 1 {
                    for _ in 0..self.num_anchors {
                        centers.push(cx);
                        centers.push(cy);
                    }
                } else {
                    centers.push(cx);
                    centers.push(cy);
                }
            }
        }
        
        let n_points = (height * width * self.num_anchors as i32) as usize;
        Array2::from_shape_vec((n_points, 2), centers)
            .expect("Failed to create anchor centers array")
    }
    
    /// Convert distance predictions to bounding boxes (array version)
    fn distance_to_bbox_array(
        &self,
        points: &Array2<f32>,
        distances: &Array2<f32>,
        max_shape: Option<(i32, i32)>,
    ) -> Array2<f32> {
        let n_boxes = points.shape()[0];
        let mut boxes = Array2::zeros((n_boxes, 4));
        
        for i in 0..n_boxes {
            let cx = points[[i, 0]];
            let cy = points[[i, 1]];
            
            let x1 = cx - distances[[i, 0]];
            let y1 = cy - distances[[i, 1]];
            let x2 = cx + distances[[i, 2]];
            let y2 = cy + distances[[i, 3]];
            
            if let Some((max_w, max_h)) = max_shape {
                boxes[[i, 0]] = x1.max(0.0).min(max_w as f32);
                boxes[[i, 1]] = y1.max(0.0).min(max_h as f32);
                boxes[[i, 2]] = x2.max(0.0).min(max_w as f32);
                boxes[[i, 3]] = y2.max(0.0).min(max_h as f32);
            } else {
                boxes[[i, 0]] = x1;
                boxes[[i, 1]] = y1;
                boxes[[i, 2]] = x2;
                boxes[[i, 3]] = y2;
            }
        }
        
        boxes
    }
    
    /// Postprocess model outputs to get final detections
    fn postprocess(
        &self,
        scores_list: Vec<Array1<f32>>,
        bboxes_list: Vec<Array2<f32>>,
        kpss_list: Vec<Array3<f32>>,
        det_scale: f32,
        _img_shape: (i32, i32),
    ) -> Result<Vec<FaceDetection>> {
        // Concatenate all detections
        let all_scores = self.concatenate_1d(&scores_list)?;
        let all_bboxes = self.concatenate_2d(&bboxes_list)?;
        
        // Sort by score
        let mut indices: Vec<usize> = (0..all_scores.len()).collect();
        indices.sort_by(|&a, &b| {
            all_scores[b].partial_cmp(&all_scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Scale bboxes back to original image size
        let scaled_bboxes: Array2<f32> = &all_bboxes / det_scale;
        
        // Apply NMS
        let keep = self.nms(&scaled_bboxes, &all_scores, &indices)?;
        
        // Collect final detections
        let mut detections = Vec::new();
        
        for &idx in &keep {
            let orig_idx = indices[idx];
            let bbox = Rect::new(
                scaled_bboxes[[orig_idx, 0]] as i32,
                scaled_bboxes[[orig_idx, 1]] as i32,
                (scaled_bboxes[[orig_idx, 2]] - scaled_bboxes[[orig_idx, 0]]) as i32,
                (scaled_bboxes[[orig_idx, 3]] - scaled_bboxes[[orig_idx, 1]]) as i32,
            );
            
            let score = all_scores[orig_idx];
            
            // Handle keypoints if available
            let keypoints = if self.with_kps && !kpss_list.is_empty() {
                let all_kpss = self.concatenate_3d(&kpss_list)?;
                let kps = all_kpss.slice(s![orig_idx, .., ..]);
                Some(
                    (0..5)
                        .map(|i| Point2f::new(
                            kps[[i, 0]] / det_scale,
                            kps[[i, 1]] / det_scale,
                        ))
                        .collect(),
                )
            } else {
                None
            };
            
            detections.push(FaceDetection {
                bbox,
                score,
                keypoints,
            });
        }
        
        Ok(detections)
    }
    
    /// Non-Maximum Suppression
    fn nms(
        &self,
        bboxes: &Array2<f32>,
        _scores: &Array1<f32>,
        order: &[usize],
    ) -> Result<Vec<usize>> {
        let mut keep = Vec::new();
        let mut order = order.to_vec();
        
        while !order.is_empty() {
            let i = order[0];
            keep.push(i);
            
            if order.len() == 1 {
                break;
            }
            
            let x1_i = bboxes[[i, 0]];
            let y1_i = bboxes[[i, 1]];
            let x2_i = bboxes[[i, 2]];
            let y2_i = bboxes[[i, 3]];
            let area_i = (x2_i - x1_i + 1.0) * (y2_i - y1_i + 1.0);
            
            let mut remaining = Vec::new();
            
            for &j in order.iter().skip(1) {
                let x1_j = bboxes[[j, 0]];
                let y1_j = bboxes[[j, 1]];
                let x2_j = bboxes[[j, 2]];
                let y2_j = bboxes[[j, 3]];
                let area_j = (x2_j - x1_j + 1.0) * (y2_j - y1_j + 1.0);
                
                let x1 = x1_i.max(x1_j);
                let y1 = y1_i.max(y1_j);
                let x2 = x2_i.min(x2_j);
                let y2 = y2_i.min(y2_j);
                
                let w = (x2 - x1 + 1.0).max(0.0);
                let h = (y2 - y1 + 1.0).max(0.0);
                let inter = w * h;
                
                let iou = inter / (area_i + area_j - inter);
                
                if iou <= self.nms_threshold {
                    remaining.push(j);
                }
            }
            
            order = remaining;
        }
        
        Ok(keep)
    }
    
    /// Helper function to concatenate 1D arrays
    fn concatenate_1d(&self, arrays: &[Array1<f32>]) -> Result<Array1<f32>> {
        if arrays.is_empty() {
            return Ok(Array1::zeros(0));
        }
        
        let total_len: usize = arrays.iter().map(|a| a.len()).sum();
        let mut result = Array1::zeros(total_len);
        
        let mut offset = 0;
        for array in arrays {
            let len = array.len();
            result.slice_mut(s![offset..offset + len]).assign(array);
            offset += len;
        }
        
        Ok(result)
    }
    
    /// Helper function to concatenate 2D arrays along axis 0
    fn concatenate_2d(&self, arrays: &[Array2<f32>]) -> Result<Array2<f32>> {
        if arrays.is_empty() {
            return Ok(Array2::zeros((0, 4)));
        }
        
        let total_rows: usize = arrays.iter().map(|a| a.shape()[0]).sum();
        let cols = arrays[0].shape()[1];
        let mut result = Array2::zeros((total_rows, cols));
        
        let mut offset = 0;
        for array in arrays {
            let rows = array.shape()[0];
            result.slice_mut(s![offset..offset + rows, ..]).assign(array);
            offset += rows;
        }
        
        Ok(result)
    }
    
    /// Helper function to concatenate 3D arrays along axis 0
    fn concatenate_3d(&self, arrays: &[Array3<f32>]) -> Result<Array3<f32>> {
        if arrays.is_empty() {
            return Ok(Array3::zeros((0, 5, 2)));
        }
        
        let total_rows: usize = arrays.iter().map(|a| a.shape()[0]).sum();
        let (_, d1, d2) = arrays[0].dim();
        let mut result = Array3::zeros((total_rows, d1, d2));
        
        let mut offset = 0;
        for array in arrays {
            let rows = array.shape()[0];
            result.slice_mut(s![offset..offset + rows, .., ..]).assign(array);
            offset += rows;
        }
        
        Ok(result)
    }

    /// Convert distance predictions to keypoints (array version)
    fn distance_to_kps_array(
        &self,
        points: &Array2<f32>,
        distances: &[f32],
        max_shape: Option<(i32, i32)>,
    ) -> Result<Array3<f32>> {
        let n_points = points.shape()[0];
        let n_kps = 5; // 5 keypoints for face
        let mut kpss = Array3::zeros((n_points, n_kps, 2));
        
        for i in 0..n_points {
            let cx = points[[i, 0]];
            let cy = points[[i, 1]];
            
            for j in 0..n_kps {
                let idx = i * (n_kps * 2) + j * 2;
                if idx + 1 >= distances.len() {
                    break;
                }
                
                let px = cx + distances[idx];
                let py = cy + distances[idx + 1];
                
                if let Some((max_w, max_h)) = max_shape {
                    kpss[[i, j, 0]] = px.max(0.0).min(max_w as f32);
                    kpss[[i, j, 1]] = py.max(0.0).min(max_h as f32);
                } else {
                    kpss[[i, j, 0]] = px;
                    kpss[[i, j, 1]] = py;
                }
            }
        }
        
        Ok(kpss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_bbox() {
        let points = vec![(100.0, 100.0), (200.0, 200.0)];
        let distances = vec![
            10.0, 10.0, 20.0, 20.0, // First box
            15.0, 15.0, 25.0, 25.0, // Second box
        ];

        let boxes = FaceDetector::distance_to_bbox(&points, &distances, Some((640, 480)));

        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].x, 90);
        assert_eq!(boxes[0].y, 90);
        assert_eq!(boxes[0].width, 30);
        assert_eq!(boxes[0].height, 30);
    }
}
