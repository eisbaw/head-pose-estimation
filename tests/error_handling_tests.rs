//! Error handling tests for all modules

use head_pose_estimation::{
    error::{AppError, Result},
    filters::create_filter,
    movement_detector::MovementDetector,
    utils::{refine_boxes, safe_cast::*},
};
use opencv::core::Rect;

#[test]
fn test_filter_creation_errors() {
    // Test invalid filter type
    let result = create_filter("invalid_filter");
    assert!(result.is_err());
    
    // Test invalid window size for moving average
    let result = create_filter("movingaverage:0");
    assert!(result.is_err());
    match result {
        Err(AppError::FilterError(msg)) => assert!(msg.contains("Window size")),
        _ => panic!("Expected FilterError"),
    }
    
    // Test invalid alpha for exponential filter
    let result = create_filter("exponential:2.0");
    assert!(result.is_err());
    match result {
        Err(AppError::FilterError(msg)) => assert!(msg.contains("Alpha")),
        _ => panic!("Expected FilterError"),
    }
    
    // Test invalid alpha for exponential filter (negative)
    let result = create_filter("exponential:-0.1");
    assert!(result.is_err());
    
    // Test invalid window size for median filter
    let result = create_filter("median:0");
    assert!(result.is_err());
    
    // Test Kalman filter creation (should always succeed)
    let result = create_filter("kalman");
    assert!(result.is_ok());
}

#[test]
fn test_movement_detector_edge_cases() {
    // Test with zero window size (should not panic, just create empty detector)
    let mut detector = MovementDetector::new(0, 2.0);
    assert!(!detector.update(1.0, 1.0)); // Should return false
    
    // Test with negative threshold (should work, just never detect movement)
    let mut detector = MovementDetector::new(5, -1.0);
    for i in 0..10 {
        detector.update(i as f64, i as f64);
    }
    // With negative threshold, movement is always detected when window is full
    
    // Test with very large window
    let mut detector = MovementDetector::new(1000, 2.0);
    assert!(!detector.update(1.0, 1.0)); // Not enough data
}

#[test] 
fn test_invalid_filter_parameters() {
    // Test invalid parameters that should cause errors
    let test_cases = vec![
        "movingaverage:-5", // Negative window size
        "median:abc", // Non-numeric parameter
        "exponential:10", // Alpha > 1
        "lowpass:-1", // Negative cutoff
        "lowpass:0", // Zero cutoff
        "hampel:5:-1", // Negative threshold
    ];
    
    for filter_str in test_cases {
        let result = create_filter(filter_str);
        assert!(result.is_err(), "Expected error for {}", filter_str);
    }
}

#[test]
fn test_safe_cast_errors() {
    // Test usize overflow
    if std::mem::size_of::<usize>() > 4 {
        let large_value = (i32::MAX as usize) + 1;
        assert!(usize_to_i32(large_value).is_err());
    }
    
    // Test u32 overflow
    let large_u32 = u32::MAX;
    assert!(u32_to_i32(large_u32).is_err());
    
    // Test f32 non-finite values
    assert!(f32_to_i32(f32::NAN).is_err());
    assert!(f32_to_i32(f32::INFINITY).is_err());
    assert!(f32_to_i32(f32::NEG_INFINITY).is_err());
    
    // Test f64 non-finite values
    assert!(f64_to_i32(f64::NAN).is_err());
    assert!(f64_to_i32(f64::INFINITY).is_err());
    assert!(f64_to_i32(f64::NEG_INFINITY).is_err());
    
    // Test f64 out of range
    assert!(f64_to_i32(1e10).is_err());
    assert!(f64_to_i32(-1e10).is_err());
}

#[test]
fn test_refine_boxes_error_handling() {
    // Test with empty boxes (should succeed)
    let mut empty_boxes = vec![];
    let result = refine_boxes(&mut empty_boxes, 100, 100, 0.0);
    assert!(result.is_ok());
    
    // Test with normal boxes
    let mut boxes = vec![Rect::new(10, 10, 50, 50)];
    let result = refine_boxes(&mut boxes, 100, 100, 10.0);
    assert!(result.is_ok());
}

#[test]
fn test_filter_chain_extreme_values() {
    // Create a Hampel filter
    let filter = create_filter("hampel:5:3.0");
    assert!(filter.is_ok());
    
    // Test with extreme values that might cause issues
    let mut filter = filter.unwrap();
    let extreme_values = vec![
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        (f64::NAN, f64::NAN),
        (1e308, -1e308),
        (-1e308, 1e308),
    ];
    
    for (pitch, yaw) in extreme_values {
        let (result_pitch, result_yaw) = filter.apply(pitch, yaw);
        // Hampel filter might return the input value if history is not full
        // Just make sure it doesn't panic
        let _ = (result_pitch, result_yaw);
    }
}

#[test]
fn test_concurrent_error_handling() {
    use std::sync::Arc;
    use std::thread;
    
    // Test thread safety of error types
    let error = Arc::new(AppError::InvalidInput("Test error".to_string()));
    
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let error_clone = Arc::clone(&error);
            thread::spawn(move || {
                let msg = format!("{}", error_clone);
                assert!(msg.contains("Test error"));
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_error_display_formatting() {
    let errors = vec![
        AppError::InvalidInput("Test input error".to_string()),
        AppError::ModelError("Test model error".to_string()),
        AppError::CursorControl("Test cursor error".to_string()),
        AppError::FilterError("Test filter error".to_string()),
    ];
    
    for error in errors {
        let display = format!("{}", error);
        assert!(!display.is_empty());
        assert!(display.contains("Test"));
    }
}

#[test]
fn test_filter_parameter_parsing() {
    // Test various parameter formats
    let test_cases = vec![
        ("movingaverage:10", true),
        ("movingaverage:abc", false),
        ("exponential:0.5", true),
        ("exponential:1.5", false),
        ("median:5", true),
        ("median:-5", false),
        ("lowpass:0.5", true),
        ("secondorderlowpass:30:0.707", true),
        ("hampel:5:3.0", true),
        ("hampel:0:3.0", false),
    ];
    
    for (filter_str, should_succeed) in test_cases {
        let result = create_filter(filter_str);
        assert_eq!(result.is_ok(), should_succeed, "Failed for {}", filter_str);
    }
}

#[test]
fn test_error_conversion_traits() {
    // Test that our error types implement necessary traits
    let error = AppError::InvalidInput("Test".to_string());
    
    // Test Display
    let _display = format!("{}", error);
    
    // Test Debug
    let _debug = format!("{:?}", error);
    
    // Test Send + Sync (implicitly tested by thread test above)
}

#[test]
fn test_result_type_operations() {
    // Test Result type operations
    let ok_result: Result<i32> = Ok(42);
    let err_result: Result<i32> = Err(AppError::InvalidInput("Test".to_string()));
    
    // Test is_ok and is_err
    assert!(ok_result.is_ok());
    assert!(!ok_result.is_err());
    assert!(!err_result.is_ok());
    assert!(err_result.is_err());
    
    // Test map operations
    let mapped_ok = ok_result.map(|x| x * 2);
    assert_eq!(mapped_ok.unwrap(), 84);
    
    let mapped_err = err_result.map(|x| x * 2);
    assert!(mapped_err.is_err());
}