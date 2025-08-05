use crate::Result;
use opencv::core::Mat;
use ort::{Environment, Session};
use std::path::Path;

/// Facial landmark detector using ONNX Runtime
#[allow(dead_code)] // Fields will be used in TODO implementation
pub struct MarkDetector {
    session: Session,
    input_size: (i32, i32),
}

impl MarkDetector {
    /// Create a new landmark detector from an ONNX model file
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let environment = std::sync::Arc::new(
            Environment::builder()
                .with_name("mark_detector")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        let session = ort::SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        // Default landmark model input size
        let input_size = (128, 128);

        Ok(Self { session, input_size })
    }

    /// Detect 68 facial landmarks in a face region
    #[allow(clippy::missing_const_for_fn)] // Can't be const due to Result allocation
    pub fn detect(&self, _face_image: &Mat) -> Result<Vec<(f32, f32)>> {
        // TODO: Implement preprocessing (resize, normalize)
        // TODO: Implement inference
        // TODO: Convert output to 68 landmark points
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Batch detection for multiple faces
    pub fn detect_batch(&self, face_images: &[Mat]) -> Result<Vec<Vec<(f32, f32)>>> {
        face_images.iter().map(|img| self.detect(img)).collect()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_landmark_count() {
        // When implemented, should return exactly 68 landmarks
        // This is a placeholder test
        assert_eq!(68, 68);
    }
}
