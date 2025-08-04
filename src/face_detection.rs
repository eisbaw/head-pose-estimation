use opencv::{core::{Mat, Rect}};
use ort::{Environment, Session};
use std::path::Path;
use crate::Result;

/// SCRFD Face Detector using ONNX Runtime
pub struct FaceDetector {
    session: Session,
    input_size: (i32, i32),
    conf_threshold: f32,
    nms_threshold: f32,
}

impl FaceDetector {
    /// Create a new face detector from an ONNX model file
    pub fn new<P: AsRef<Path>>(model_path: P, conf_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let environment = std::sync::Arc::new(
            Environment::builder()
                .with_name("face_detector")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?
        );
            
        let session = ort::SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;
            
        // Default SCRFD input size
        let input_size = (640, 640);
        
        Ok(Self {
            session,
            input_size,
            conf_threshold,
            nms_threshold,
        })
    }
    
    /// Detect faces in an image
    pub fn detect(&self, _image: &Mat) -> Result<Vec<Rect>> {
        // TODO: Implement preprocessing
        // TODO: Implement inference
        // TODO: Implement postprocessing with NMS
        // For now, return empty vector
        Ok(Vec::new())
    }
    
    /// Convert distance predictions to bounding boxes
    fn distance_to_bbox(points: &[(f32, f32)], distances: &[f32], max_shape: Option<(i32, i32)>) -> Vec<Rect> {
        // Port of distance2bbox from Python
        let mut boxes = Vec::new();
        
        for (i, &(cx, cy)) in points.iter().enumerate() {
            if i * 4 + 3 >= distances.len() {
                break;
            }
            
            let x1 = cx - distances[i * 4];
            let y1 = cy - distances[i * 4 + 1];
            let x2 = cx + distances[i * 4 + 2];
            let y2 = cy + distances[i * 4 + 3];
            
            let mut bbox = Rect::new(
                x1 as i32,
                y1 as i32,
                (x2 - x1) as i32,
                (y2 - y1) as i32,
            );
            
            // Clip to image boundaries if max_shape is provided
            if let Some((max_w, max_h)) = max_shape {
                bbox.x = bbox.x.max(0);
                bbox.y = bbox.y.max(0);
                bbox.width = bbox.width.min(max_w - bbox.x);
                bbox.height = bbox.height.min(max_h - bbox.y);
            }
            
            boxes.push(bbox);
        }
        
        boxes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_bbox() {
        let points = vec![(100.0, 100.0), (200.0, 200.0)];
        let distances = vec![
            10.0, 10.0, 20.0, 20.0,  // First box
            15.0, 15.0, 25.0, 25.0,  // Second box
        ];
        
        let boxes = FaceDetector::distance_to_bbox(&points, &distances, Some((640, 480)));
        
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].x, 90);
        assert_eq!(boxes[0].y, 90);
        assert_eq!(boxes[0].width, 30);
        assert_eq!(boxes[0].height, 30);
    }
}