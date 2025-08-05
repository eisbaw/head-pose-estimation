//! Integration tests for the head pose estimation pipeline

use head_pose_estimation::{
    face_detection::FaceDetector,
    filters::create_filter,
    mark_detection::MarkDetector,
    movement_detector::MovementDetector,
    pose_estimation::PoseEstimator,
};
use opencv::{core::Mat, prelude::*};

/// Test the complete pipeline from image to pose estimation
#[test]
#[ignore = "Requires ONNX models and test images"]
fn test_full_pipeline() {
    // Initialize components
    let mut face_detector = FaceDetector::new("assets/face_detector.onnx", 0.7, 0.3).expect("Failed to create face detector");
    let mark_detector = MarkDetector::new("assets/face_landmarks.onnx").expect("Failed to create mark detector");
    let pose_estimator = PoseEstimator::new("assets/model.txt", 800, 600).expect("Failed to create pose estimator");
    
    // Create a synthetic test image (640x480 black image)
    let test_image = Mat::zeros(480, 640, opencv::core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();
    
    // Detect faces
    let faces = face_detector.detect(&test_image).expect("Face detection failed");
    
    // For each detected face, detect landmarks and estimate pose
    for _face in faces {
        // For testing, use the whole image instead of extracting the face region
        // In a real scenario, you would extract the face region first
        let landmarks = mark_detector
            .detect(&test_image)
            .expect("Landmark detection failed");
        
        // Convert landmarks to tuples for pose estimation
        let landmark_tuples: Vec<(f32, f32)> = landmarks.iter()
            .map(|p| (p.x, p.y))
            .collect();
        
        let (rotation_vector, translation_vector, _) = pose_estimator
            .estimate_pose(&landmark_tuples)
            .expect("Pose estimation failed");
        
        // Verify outputs are reasonable (Vec3d has 3 elements)
        // rotation_vector and translation_vector are Vec3d (3-element vectors)
        // Just verify they contain finite values
        assert!(rotation_vector.get(0).unwrap().is_finite());
        assert!(rotation_vector.get(1).unwrap().is_finite());
        assert!(rotation_vector.get(2).unwrap().is_finite());
        assert!(translation_vector.get(0).unwrap().is_finite());
        assert!(translation_vector.get(1).unwrap().is_finite());
        assert!(translation_vector.get(2).unwrap().is_finite());
    }
}

/// Test filter integration with pose data
#[test]
fn test_filter_pipeline() {
    // Create different filter types
    let filter_names = vec![
        "none",
        "moving_average",
        "median",
        "exponential",
        "kalman",
        "lowpass",
        "lowpass2",
        "hampel",
    ];
    
    for filter_name in filter_names {
        let mut filter = create_filter(filter_name).expect("Failed to create filter");
        
        // Simulate pose data stream
        let test_data = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (10.0, 10.0), // Outlier
            (4.0, 4.0),
            (5.0, 5.0),
        ];
        
        for &(pitch, yaw) in &test_data {
            let filtered = filter.apply(pitch, yaw);
            
            // Verify filtered values are finite
            assert!(filtered.0.is_finite(), "Filter {} produced non-finite pitch", filter_name);
            assert!(filtered.1.is_finite(), "Filter {} produced non-finite yaw", filter_name);
        }
        
        // Test reset functionality
        filter.reset();
        let (pitch, yaw) = filter.apply(0.0, 0.0);
        assert!(pitch.is_finite());
        assert!(yaw.is_finite());
    }
}

/// Test movement detection with simulated pose data
#[test]
fn test_movement_detection_integration() {
    let mut detector = MovementDetector::new(30, 1.0);
    
    // Simulate still pose
    for i in 0..60 {
        let is_moving = detector.update(0.0, 0.0);
        // After filling the window (30 samples), it should detect no movement
        if i >= 30 {
            assert!(!is_moving, "Should not detect movement for still pose after window is filled");
        }
    }
    
    // Reset and simulate movement
    detector = MovementDetector::new(30, 1.0);
    let mut angle: f64 = 0.0;
    let mut detected_movement = false;
    for _ in 0..60 {
        angle += 2.0;
        let is_moving = detector.update(angle.sin() * 10.0, angle.cos() * 10.0);
        if is_moving {
            detected_movement = true;
        }
    }
    assert!(detected_movement, "Should detect movement for oscillating pose");
}

/// Test the complete pipeline with multiple filters
#[test]
fn test_multi_filter_comparison() {
    // Create multiple filters for comparison
    let mut filters: Vec<Box<dyn head_pose_estimation::filters::CursorFilter>> = vec![
        create_filter("none").unwrap(),
        create_filter("moving_average").unwrap(),
        create_filter("kalman").unwrap(),
        create_filter("lowpass").unwrap(),
    ];
    
    // Simulate noisy pose data
    let mut rng = 0u32;
    let test_duration = 100;
    let mut results = Vec::new();
    
    for i in 0..test_duration {
        // Simple linear congruential generator for deterministic "random" noise
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng / 65536) % 1000) as f64 / 1000.0 - 0.5;
        
        let true_pitch = (i as f64 * 0.1).sin() * 30.0;
        let true_yaw = (i as f64 * 0.1).cos() * 30.0;
        
        let noisy_pitch = true_pitch + noise * 5.0;
        let noisy_yaw = true_yaw + noise * 5.0;
        
        let mut frame_results = Vec::new();
        for filter in &mut filters {
            let (filtered_pitch, filtered_yaw) = filter.apply(noisy_pitch, noisy_yaw);
            frame_results.push((filtered_pitch, filtered_yaw));
        }
        results.push(frame_results);
    }
    
    // Verify all filters produced valid output
    for frame_results in &results {
        for &(pitch, yaw) in frame_results {
            assert!(pitch.is_finite());
            assert!(yaw.is_finite());
            assert!(pitch.abs() < 100.0, "Pitch value out of reasonable range");
            assert!(yaw.abs() < 100.0, "Yaw value out of reasonable range");
        }
    }
    
    // Verify filtered values are smoother than unfiltered (except NoFilter)
    let mut variances = Vec::new();
    for filter_idx in 0..filters.len() {
        let values: Vec<f64> = results.iter()
            .map(|frame| frame[filter_idx].0) // Use pitch values
            .collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variances.push(variance);
    }
    
    // NoFilter (index 0) should have higher variance than filtered versions
    let no_filter_variance = variances[0];
    for (i, &variance) in variances.iter().enumerate().skip(1) {
        assert!(
            variance <= no_filter_variance * 1.1, // Allow 10% tolerance
            "Filter at index {} should reduce variance (no_filter: {}, filtered: {})",
            i, no_filter_variance, variance
        );
    }
}

/// Test error handling in the pipeline
#[test]
fn test_pipeline_error_handling() {
    // Test with invalid model paths
    let face_detector_result = FaceDetector::new("nonexistent_model.onnx", 0.7, 0.3);
    assert!(face_detector_result.is_err(), "Should fail with invalid model path");
    
    let mark_detector_result = MarkDetector::new("nonexistent_model.onnx");
    assert!(mark_detector_result.is_err(), "Should fail with invalid model path");
    
    // Test pose estimator with edge case dimensions
    let pose_estimator_zero = PoseEstimator::new("assets/model.txt", 0, 0);
    assert!(pose_estimator_zero.is_ok(), "Should handle zero dimensions gracefully");
    
    let pose_estimator_large = PoseEstimator::new("assets/model.txt", 10000, 10000);
    assert!(pose_estimator_large.is_ok(), "Should handle large dimensions");
}

/// Test thread safety of filters
#[test]
fn test_filter_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let filter = Arc::new(Mutex::new(create_filter("moving_average").unwrap()));
    let mut handles = vec![];
    
    // Spawn multiple threads that use the filter
    for thread_id in 0..4 {
        let filter_clone = Arc::clone(&filter);
        let handle = thread::spawn(move || {
            for i in 0..25 {
                let value = (thread_id * 25 + i) as f64;
                let mut filter = filter_clone.lock().unwrap();
                let (pitch, yaw) = filter.apply(value, value);
                assert!(pitch.is_finite());
                assert!(yaw.is_finite());
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}