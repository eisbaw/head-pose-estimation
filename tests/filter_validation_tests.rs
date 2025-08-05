//! Tests for filter parameter validation

use head_pose_estimation::filters::{
    create_filter,
    exponential::ExponentialFilter,
    hampel::HampelFilter,
    low_pass::{LowPassFilter, SecondOrderLowPassFilter},
    median::MedianFilter,
    moving_average::MovingAverageFilter,
};

#[test]
#[should_panic(expected = "Window size must be greater than 0")]
fn test_moving_average_zero_window() {
    let _ = MovingAverageFilter::new(0);
}

#[test]
#[should_panic(expected = "Window size must be greater than 0")]
fn test_median_zero_window() {
    let _ = MedianFilter::new(0);
}

#[test]
#[should_panic(expected = "Median filter window size must be odd")]
fn test_median_even_window() {
    let _ = MedianFilter::new(4);
}

#[test]
#[should_panic(expected = "Alpha must be in (0, 1]")]
fn test_exponential_zero_alpha() {
    let _ = ExponentialFilter::new(0.0);
}

#[test]
#[should_panic(expected = "Alpha must be in (0, 1]")]
fn test_exponential_too_large_alpha() {
    let _ = ExponentialFilter::new(1.5);
}

#[test]
#[should_panic(expected = "Alpha must be in (0, 1]")]
fn test_low_pass_zero_alpha() {
    let _ = LowPassFilter::new(0.0);
}

#[test]
#[should_panic(expected = "Cutoff frequency must be positive")]
fn test_second_order_low_pass_zero_cutoff() {
    let _ = SecondOrderLowPassFilter::new(0.0, 0.7);
}

#[test]
#[should_panic(expected = "Damping ratio must be positive")]
fn test_second_order_low_pass_zero_damping() {
    let _ = SecondOrderLowPassFilter::new(10.0, 0.0);
}

#[test]
#[should_panic(expected = "Window size must be greater than 0")]
fn test_hampel_zero_window() {
    let _ = HampelFilter::new(0, 3.0);
}

#[test]
#[should_panic(expected = "Window size must be odd")]
fn test_hampel_even_window() {
    let _ = HampelFilter::new(6, 3.0);
}

#[test]
#[should_panic(expected = "Threshold must be non-negative")]
fn test_hampel_negative_threshold() {
    let _ = HampelFilter::new(5, -1.0);
}

#[test]
fn test_create_filter_validation() {
    // Test that create_filter validates parameters
    assert!(create_filter("movingaverage:0").is_err());
    assert!(create_filter("median:0").is_err());
    assert!(create_filter("median:4").is_err()); // create_filter now enforces odd window
    assert!(create_filter("exponential:0").is_err());
    assert!(create_filter("exponential:1.5").is_err());
    assert!(create_filter("lowpass:0").is_err());
    assert!(create_filter("lowpass:1.5").is_err());
    assert!(create_filter("hampel:0:3").is_err());
    assert!(create_filter("hampel:5:-1").is_err());

    // Valid parameters should work
    assert!(create_filter("movingaverage:5").is_ok());
    assert!(create_filter("median:5").is_ok());
    assert!(create_filter("exponential:0.5").is_ok());
    assert!(create_filter("lowpass:0.3").is_ok());
    assert!(create_filter("secondorderlowpass:10:0.7").is_ok());
    assert!(create_filter("hampel:7:3").is_ok());
}

#[test]
fn test_filter_handles_edge_values() {
    // Test filters with edge case values
    let test_cases = vec![
        ("kalman", vec![f64::NAN, f64::INFINITY, -f64::INFINITY]),
        ("movingaverage:3", vec![f64::NAN, 0.0, f64::INFINITY]),
        ("median:3", vec![f64::NAN, f64::INFINITY, 1.0]),
        ("exponential:0.5", vec![f64::NAN, f64::INFINITY, -f64::INFINITY]),
    ];

    for (filter_type, values) in test_cases {
        let mut filter = create_filter(filter_type).unwrap();

        // Apply edge values and ensure no panic
        for &val in &values {
            let (pitch, yaw) = filter.apply(val, val);
            // Just verify we don't panic - actual behavior with NaN/Inf may vary
            let _ = (pitch, yaw);
        }
    }
}
