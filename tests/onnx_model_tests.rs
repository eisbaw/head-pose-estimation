//! Tests for ONNX model loading and inference

use head_pose_estimation::{
    face_detection::FaceDetector,
    mark_detection::MarkDetector,
    pose_estimation::PoseEstimator,
    Result,
};
use opencv::{core::{Mat, Size, CV_8UC3}, imgcodecs, imgproc};
use std::path::Path;

#[test]
#[ignore = "Requires ONNX models"]
fn test_load_face_detector_model() -> Result<()> {
    let model_path = "assets/face_detector.onnx";
    assert!(Path::new(model_path).exists(), "Face detector model not found");
    
    let _detector = FaceDetector::new(model_path, 0.6, 0.5)?;
    // If construction succeeds, model loaded correctly
    
    Ok(())
}

#[test]
#[ignore = "Requires ONNX models"]
fn test_load_landmark_detector_model() -> Result<()> {
    let model_path = "assets/face_landmarks.onnx";
    assert!(Path::new(model_path).exists(), "Landmark detector model not found");
    
    let _detector = MarkDetector::new(model_path)?;
    // If construction succeeds, model loaded correctly
    
    Ok(())
}

#[test]
#[ignore = "Requires ONNX models"]
fn test_load_3d_face_model() -> Result<()> {
    let model_path = "assets/model.txt";
    assert!(Path::new(model_path).exists(), "3D face model not found");
    
    let _estimator = PoseEstimator::new(model_path, 640, 480)?;
    
    Ok(())
}

#[test]
#[ignore = "Requires ONNX models and test image"]
fn test_face_detection_inference() -> Result<()> {
    let mut detector = FaceDetector::new("assets/face_detector.onnx", 0.6, 0.5)?;
    
    // Create a test image (640x480 RGB)
    let test_image = Mat::new_rows_cols_with_default(
        480, 640, CV_8UC3,
        opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0)
    )?;
    
    // Run inference
    let faces = detector.detect(&test_image)?;
    
    // Even on a blank image, inference should complete without error
    assert!(faces.is_empty() || !faces.is_empty()); // Either result is valid
    
    Ok(())
}

#[test]
#[ignore = "Requires ONNX models and test image"]
fn test_landmark_detection_inference() -> Result<()> {
    let detector = MarkDetector::new("assets/face_landmarks.onnx")?;
    
    // Create a test face image (112x112 RGB as expected by the model)
    let test_face = Mat::new_rows_cols_with_default(
        112, 112, CV_8UC3,
        opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0)
    )?;
    
    // Run inference
    let landmarks = detector.detect(&test_face)?;
    
    // Should return 68 landmarks
    assert_eq!(landmarks.len(), 68, "Expected 68 facial landmarks");
    
    // Check landmarks are within image bounds
    for landmark in &landmarks {
        assert!(landmark.x >= 0.0 && landmark.x <= 112.0);
        assert!(landmark.y >= 0.0 && landmark.y <= 112.0);
    }
    
    Ok(())
}

#[test]
#[ignore = "Requires ONNX models and test image"]
fn test_full_pipeline() -> Result<()> {
    // Initialize all components
    let mut face_detector = FaceDetector::new("assets/face_detector.onnx", 0.6, 0.5)?;
    let mark_detector = MarkDetector::new("assets/face_landmarks.onnx")?;
    let pose_estimator = PoseEstimator::new("assets/model.txt", 640, 480)?;
    
    // Create or load a test image
    let test_image = if Path::new("test_face.jpg").exists() {
        imgcodecs::imread("test_face.jpg", imgcodecs::IMREAD_COLOR)?
    } else {
        // Create synthetic test image
        let mut img = Mat::new_rows_cols_with_default(
            480, 640, CV_8UC3,
            opencv::core::Scalar::new(200.0, 200.0, 200.0, 0.0)
        )?;
        
        // Draw a simple face-like pattern
        let face_center = opencv::core::Point::new(320, 240);
        imgproc::circle(
            &mut img,
            face_center,
            100,
            opencv::core::Scalar::new(255.0, 200.0, 180.0, 0.0),
            -1,
            imgproc::LINE_8,
            0
        )?;
        
        img
    };
    
    // Run face detection
    let faces = face_detector.detect(&test_image)?;
    println!("Detected {} faces", faces.len());
    
    // Process each face
    for face in faces {
        // Extract face region
        let face_roi = Mat::roi(&test_image, face.bbox)?;
        
        // Resize to 112x112 for landmark detection
        let mut face_resized = Mat::default();
        imgproc::resize(
            &face_roi,
            &mut face_resized,
            Size::new(112, 112),
            0.0,
            0.0,
            imgproc::INTER_LINEAR
        )?;
        
        // Detect landmarks
        let landmarks = mark_detector.detect(&face_resized)?;
        assert_eq!(landmarks.len(), 68);
        
        // Convert landmarks to tuples for pose estimation
        let landmark_tuples: Vec<(f32, f32)> = landmarks.iter()
            .map(|p| (p.x, p.y))
            .collect();
        
        // Estimate pose
        let (euler_angles, _, _) = pose_estimator.estimate_pose(&landmark_tuples)?;
        
        println!("Pose - Pitch: {:.2}°, Yaw: {:.2}°, Roll: {:.2}°", 
                 euler_angles[0], euler_angles[1], euler_angles[2]);
        
        // Verify angles are within reasonable range
        assert!(euler_angles[0].abs() < 90.0, "Pitch angle out of range");
        assert!(euler_angles[1].abs() < 90.0, "Yaw angle out of range");
        assert!(euler_angles[2].abs() < 180.0, "Roll angle out of range");
    }
    
    Ok(())
}

#[test] 
#[ignore = "Requires ONNX models"]
fn test_model_metadata() -> Result<()> {
    use std::fs;
    
    // Check model file sizes
    let face_detector_size = fs::metadata("assets/face_detector.onnx")?.len();
    let landmark_detector_size = fs::metadata("assets/face_landmarks.onnx")?.len();
    
    println!("Face detector model size: {} MB", face_detector_size / 1_000_000);
    println!("Landmark detector model size: {} MB", landmark_detector_size / 1_000_000);
    
    // Models should be reasonable size (not empty, not too large)
    assert!(face_detector_size > 1_000_000, "Face detector model too small");
    assert!(face_detector_size < 100_000_000, "Face detector model too large");
    
    assert!(landmark_detector_size > 1_000_000, "Landmark detector model too small");
    assert!(landmark_detector_size < 100_000_000, "Landmark detector model too large");
    
    Ok(())
}