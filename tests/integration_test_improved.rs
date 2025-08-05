//! Improved integration tests with proper error handling

mod test_helpers;

use head_pose_estimation::{
    face_detection::FaceDetector, mark_detection::MarkDetector, pose_estimation::PoseEstimator, Result,
};
use test_helpers::{assert_vec3d_finite, create_test_filter, create_test_image};

/// Test the complete pipeline with proper error handling
#[test]
#[ignore = "Requires ONNX models and test images"]
fn test_full_pipeline_with_error_handling() -> Result<()> {
    // Initialize components
    let mut face_detector = FaceDetector::new("assets/face_detector.onnx", 0.7, 0.3)?;
    let mark_detector = MarkDetector::new("assets/face_landmarks.onnx")?;
    let pose_estimator = PoseEstimator::new("assets/model.txt", 800, 600)?;

    // Create a synthetic test image
    let test_image = create_test_image(480, 640, opencv::core::CV_8UC3)?;

    // Detect faces
    let faces = face_detector.detect(&test_image)?;

    // For each detected face, detect landmarks and estimate pose
    for _face in faces {
        let landmarks = mark_detector.detect(&test_image)?;

        // Convert landmarks to tuples for pose estimation
        let landmark_tuples: Vec<(f32, f32)> = landmarks.iter().map(|p| (p.x, p.y)).collect();

        let (rotation_vector, translation_vector, _) = pose_estimator.estimate_pose(&landmark_tuples)?;

        // Verify outputs are reasonable
        assert_vec3d_finite(&rotation_vector)?;
        assert_vec3d_finite(&translation_vector)?;
    }

    Ok(())
}

/// Test filter pipeline with proper error handling
#[test]
fn test_filter_pipeline_safe() -> Result<()> {
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
        let mut filter = create_test_filter(filter_name)?;

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
            if !filtered.0.is_finite() {
                return Err(head_pose_estimation::Error::FilterError(format!(
                    "Filter {filter_name} produced non-finite pitch"
                )));
            }
            if !filtered.1.is_finite() {
                return Err(head_pose_estimation::Error::FilterError(format!(
                    "Filter {filter_name} produced non-finite yaw"
                )));
            }
        }

        // Test reset functionality
        filter.reset();
        let (pitch, yaw) = filter.apply(0.0, 0.0);
        if !pitch.is_finite() || !yaw.is_finite() {
            return Err(head_pose_estimation::Error::FilterError(format!(
                "Filter {filter_name} produced non-finite values after reset"
            )));
        }
    }

    Ok(())
}

/// Test filter comparison with error handling
#[test]
fn test_multi_filter_comparison_safe() -> Result<()> {
    // Create multiple filters for comparison
    let mut filters: Vec<Box<dyn head_pose_estimation::filters::CursorFilter>> = vec![
        create_test_filter("none")?,
        create_test_filter("moving_average")?,
        create_test_filter("kalman")?,
        create_test_filter("lowpass")?,
    ];

    // Simulate noisy pose data
    let mut rng = 0u32;
    let test_duration = 100;

    for i in 0..test_duration {
        // Simple linear congruential generator for deterministic "random" noise
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng / 65536) % 1000) as f64 / 1000.0 - 0.5;

        let true_pitch = (i as f64 * 0.1).sin() * 30.0;
        let true_yaw = (i as f64 * 0.1).cos() * 30.0;

        let noisy_pitch = true_pitch + noise * 5.0;
        let noisy_yaw = true_yaw + noise * 5.0;

        for (filter_idx, filter) in filters.iter_mut().enumerate() {
            let (filtered_pitch, filtered_yaw) = filter.apply(noisy_pitch, noisy_yaw);

            if !filtered_pitch.is_finite() || !filtered_yaw.is_finite() {
                return Err(head_pose_estimation::Error::FilterError(format!(
                    "Filter {} produced non-finite values",
                    filter_idx
                )));
            }

            if filtered_pitch.abs() >= 100.0 || filtered_yaw.abs() >= 100.0 {
                return Err(head_pose_estimation::Error::FilterError(format!(
                    "Filter {} produced values out of reasonable range",
                    filter_idx
                )));
            }
        }
    }

    Ok(())
}

/// Test thread safety with proper error handling
#[test]
fn test_filter_thread_safety_safe() -> Result<()> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let filter = Arc::new(Mutex::new(create_test_filter("moving_average")?));
    let mut handles = vec![];

    // Spawn multiple threads that use the filter
    for thread_id in 0..4 {
        let filter_clone = Arc::clone(&filter);
        let handle = thread::spawn(move || -> Result<()> {
            for i in 0..25 {
                let value = (thread_id * 25 + i) as f64;
                let mut filter = filter_clone
                    .lock()
                    .map_err(|e| head_pose_estimation::Error::FilterError(format!("Failed to lock filter: {}", e)))?;
                let (pitch, yaw) = filter.apply(value, value);

                if !pitch.is_finite() || !yaw.is_finite() {
                    return Err(head_pose_estimation::Error::FilterError(
                        "Filter produced non-finite values in thread".to_string(),
                    ));
                }
            }
            Ok(())
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle
            .join()
            .map_err(|_| head_pose_estimation::Error::FilterError("Thread panicked".to_string()))??;
    }

    Ok(())
}
