use crate::{constants::NUM_FACIAL_LANDMARKS, utils::safe_cast::usize_to_i32, Result};
use ndarray::{Array1, Array4, CowArray};
use opencv::core::{Mat, Point2f, Size, CV_32F};
use opencv::imgproc::{self, InterpolationFlags};
use opencv::prelude::*;
use ort::{Environment, Session, Value};
use std::path::Path;
use std::sync::Arc;

/// Default landmark detector input size
const DEFAULT_LANDMARK_INPUT_SIZE: i32 = 128;

/// Facial landmark detector using `ONNX` Runtime
pub struct MarkDetector {
    session: Session,
    #[allow(dead_code)] // Reserved for future named tensor support
    input_name: String,
    #[allow(dead_code)] // Reserved for future named tensor support
    output_name: String,
    input_size: i32,
}

impl MarkDetector {
    /// Create a new landmark detector from an `ONNX` model file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The ONNX model file cannot be loaded
    /// - The model has an unexpected structure
    /// - The ONNX runtime environment cannot be created
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        log::info!(
            "Initializing MarkDetector with model: {}",
            model_path.as_ref().display()
        );
        let environment = Arc::new(
            Environment::builder()
                .with_name("mark_detector")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        let session = ort::SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        // Get model input/output metadata
        let input_name = session
            .inputs
            .first()
            .ok_or_else(|| crate::error::Error::ModelInputError("Model has no inputs".to_string()))?
            .name
            .clone();

        let output_name = session
            .outputs
            .first()
            .ok_or_else(|| crate::error::Error::ModelOutputError("Model has no outputs".to_string()))?
            .name
            .clone();

        // Default landmark model input size
        let input_size = DEFAULT_LANDMARK_INPUT_SIZE;

        Ok(Self {
            session,
            input_name,
            output_name,
            input_size,
        })
    }

    /// Detect facial landmarks in a face region
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch detection with a single image fails
    pub fn detect(&self, face_image: &Mat) -> Result<Vec<Point2f>> {
        let images = vec![face_image];
        let results = self.detect_batch(&images)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Batch detection for multiple faces
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image preprocessing fails
    /// - The ONNX model inference fails
    /// - The output tensor has an unexpected shape
    pub fn detect_batch(&self, face_images: &[&Mat]) -> Result<Vec<Vec<Point2f>>> {
        if face_images.is_empty() {
            return Ok(Vec::new());
        }

        // Preprocess all images
        let preprocessed = self.preprocess_batch(face_images)?;

        // Run inference
        let marks = self.forward(preprocessed)?;

        // Convert output to landmark points
        Ok(self.postprocess(&marks, face_images))
    }

    /// Preprocess a batch of images for the model
    #[allow(clippy::cast_sign_loss)] // OpenCV dimensions are positive
    fn preprocess_batch(&self, images: &[&Mat]) -> Result<Array4<f32>> {
        let batch_size = images.len();
        let size = self.input_size as usize;
        let channels = 3;

        let mut batch_data = vec![0.0f32; batch_size * size * size * channels];

        for (idx, &image) in images.iter().enumerate() {
            // Resize image
            let mut resized = Mat::default();
            imgproc::resize(
                image,
                &mut resized,
                Size::new(self.input_size, self.input_size),
                0.0,
                0.0,
                InterpolationFlags::INTER_LINEAR as i32,
            )?;

            // Convert BGR to RGB
            let mut rgb_image = Mat::default();
            imgproc::cvt_color(&resized, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;

            // Convert to f32 and normalize to [0, 1]
            let mut float_image = Mat::default();
            rgb_image.convert_to(&mut float_image, CV_32F, 1.0 / 255.0, 0.0)?;

            // Copy data to batch array
            let offset = idx * size * size * channels;
            for row in 0..size {
                for col in 0..size {
                    for ch in 0..channels {
                        let pixel =
                            float_image.at_2d::<opencv::core::Vec3f>(usize_to_i32(row)?, usize_to_i32(col)?)?[ch];
                        let batch_idx = offset + (row * size + col) * channels + ch;
                        batch_data[batch_idx] = pixel;
                    }
                }
            }
        }

        // Create NHWC array and transpose to NCHW
        let array = Array4::from_shape_vec((batch_size, size, size, channels), batch_data)
            .map_err(|e| crate::error::Error::ModelDataFormatError(format!("Failed to create array: {e}")))?;

        // Note: Some models might expect NHWC format, but most expect NCHW
        // If the model expects NCHW, uncomment the following line:
        // array = array.permuted_axes([0, 3, 1, 2]);

        Ok(array)
    }

    /// Run forward pass through the model
    fn forward(&self, inputs: Array4<f32>) -> Result<Array1<f32>> {
        // Create ONNX input
        let cow_array = CowArray::from(inputs.into_dyn());
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;

        // Run inference
        let outputs = self.session.run(vec![input_tensor])?;

        // Extract marks output
        let marks_output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| crate::error::Error::ModelOutputError("No output from model".to_string()))?;

        let marks_tensor = marks_output.try_extract::<f32>()?;
        let marks_view = marks_tensor.view();
        let marks_data = marks_view
            .as_slice()
            .ok_or_else(|| crate::error::Error::ModelOutputError("Failed to get output data".to_string()))?;

        Ok(Array1::from(marks_data.to_vec()))
    }

    /// Convert model output to landmark points
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for pixel coordinates
    fn postprocess(&self, marks: &Array1<f32>, face_images: &[&Mat]) -> Vec<Vec<Point2f>> {
        let batch_size = face_images.len();
        let n_landmarks = NUM_FACIAL_LANDMARKS;
        let n_coords = 2; // x, y

        let mut results = Vec::new();

        for (i, face_image) in face_images.iter().enumerate().take(batch_size) {
            let mut landmarks = Vec::new();
            let offset = i * n_landmarks * n_coords;

            // Get scaling factors from original face image
            let face_width = face_image.cols() as f32;
            let face_height = face_image.rows() as f32;

            for j in 0..n_landmarks {
                let idx = offset + j * n_coords;
                if idx + 1 < marks.len() {
                    // Marks are normalized to input size, scale to face image size
                    let x = marks[idx] * face_width / self.input_size as f32;
                    let y = marks[idx + 1] * face_height / self.input_size as f32;
                    landmarks.push(Point2f::new(x, y));
                }
            }

            results.push(landmarks);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_count() {
        // Test that the constant is correct
        assert_eq!(NUM_FACIAL_LANDMARKS, 68);
    }

    #[test]
    fn test_default_input_size() {
        // Test the default input size constant
        assert_eq!(DEFAULT_LANDMARK_INPUT_SIZE, 128);
    }

    #[test]
    fn test_landmark_data_structure() {
        // Test the output format expectations
        // Each landmark has 2 coordinates (x, y)
        let total_values = NUM_FACIAL_LANDMARKS * 2;
        assert_eq!(total_values, 136);
    }

    #[test]
    fn test_landmark_indices() {
        // Test specific landmark indices based on 68-point model
        // Ref: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

        // Jaw line: 0-16
        assert!(0 < NUM_FACIAL_LANDMARKS);
        assert!(16 < NUM_FACIAL_LANDMARKS);

        // Right eyebrow: 17-21
        assert!(21 < NUM_FACIAL_LANDMARKS);

        // Left eyebrow: 22-26
        assert!(26 < NUM_FACIAL_LANDMARKS);

        // Nose bridge: 27-30
        assert!(30 < NUM_FACIAL_LANDMARKS);

        // Lower nose: 31-35
        assert!(35 < NUM_FACIAL_LANDMARKS);

        // Right eye: 36-41
        assert!(41 < NUM_FACIAL_LANDMARKS);

        // Left eye: 42-47
        assert!(47 < NUM_FACIAL_LANDMARKS);

        // Outer lip: 48-59
        assert!(59 < NUM_FACIAL_LANDMARKS);

        // Inner lip: 60-67
        assert_eq!(67, NUM_FACIAL_LANDMARKS - 1);
    }
}
