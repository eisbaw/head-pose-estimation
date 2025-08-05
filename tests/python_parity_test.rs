//! Tests comparing Rust filter outputs with Python implementations
//! 
//! These tests ensure numerical accuracy matches the Python reference implementation

use head_pose_estimation::filters::create_filter;

/// Test data generated from Python filters
mod python_reference {
    /// Test sequence for filter validation
    pub const TEST_SEQUENCE: [(f64, f64); 10] = [
        (10.0, 20.0),
        (15.0, 25.0),
        (20.0, 30.0),
        (25.0, 35.0),
        (30.0, 40.0),
        (35.0, 45.0),
        (40.0, 50.0),
        (45.0, 55.0),
        (50.0, 60.0),
        (55.0, 65.0),
    ];

    /// Expected outputs from Python MovingAverageFilter(window_size=3)
    /// Note: Python casts to int, but Rust returns float for accuracy
    pub const MOVING_AVG_3_EXPECTED: [(f64, f64); 10] = [
        (10.0, 20.0),   // First value
        (12.5, 22.5),   // Average of [10, 15]
        (15.0, 25.0),   // Average of [10, 15, 20]
        (20.0, 30.0),   // Average of [15, 20, 25]
        (25.0, 35.0),   // Average of [20, 25, 30]
        (30.0, 40.0),   // Average of [25, 30, 35]
        (35.0, 45.0),   // Average of [30, 35, 40]
        (40.0, 50.0),   // Average of [35, 40, 45]
        (45.0, 55.0),   // Average of [40, 45, 50]
        (50.0, 60.0),   // Average of [45, 50, 55]
    ];

    /// Expected outputs from Python MedianFilter(window_size=3)
    pub const MEDIAN_3_EXPECTED: [(f64, f64); 10] = [
        (10.0, 20.0),   // Only one value
        (12.5, 22.5),   // Median of [10, 15] = average
        (15.0, 25.0),   // Median of [10, 15, 20]
        (20.0, 30.0),   // Median of [15, 20, 25]
        (25.0, 35.0),   // Median of [20, 25, 30]
        (30.0, 40.0),   // Median of [25, 30, 35]
        (35.0, 45.0),   // Median of [30, 35, 40]
        (40.0, 50.0),   // Median of [35, 40, 45]
        (45.0, 55.0),   // Median of [40, 45, 50]
        (50.0, 60.0),   // Median of [45, 50, 55]
    ];
}

const TOLERANCE: f64 = 0.5; // Allow 0.5 absolute difference due to Python's integer rounding

#[test]
fn test_moving_average_python_parity() {
    let mut filter = create_filter("movingaverage:3").unwrap();
    
    for (i, &(x_in, y_in)) in python_reference::TEST_SEQUENCE.iter().enumerate() {
        let (x_out, y_out) = filter.apply(x_in, y_in);
        let (x_expected, y_expected) = python_reference::MOVING_AVG_3_EXPECTED[i];
        
        assert!(
            (x_out - x_expected).abs() < TOLERANCE,
            "MovingAverage X mismatch at index {}: got {}, expected {}",
            i, x_out, x_expected
        );
        assert!(
            (y_out - y_expected).abs() < TOLERANCE,
            "MovingAverage Y mismatch at index {}: got {}, expected {}",
            i, y_out, y_expected
        );
    }
}

#[test]
fn test_exponential_filter_behavior() {
    // Test the mathematical correctness of the exponential filter
    let mut filter = create_filter("exponential:0.3").unwrap();
    
    // First value should pass through
    let (x1, y1) = filter.apply(10.0, 20.0);
    assert_eq!((x1, y1), (10.0, 20.0));
    
    // Second value: 0.3 * 15 + 0.7 * 10 = 11.5
    let (x2, y2) = filter.apply(15.0, 25.0);
    assert!((x2 - 11.5).abs() < 0.001);
    assert!((y2 - 21.5).abs() < 0.001);
    
    // Third value: 0.3 * 20 + 0.7 * 11.5 = 14.05
    let (x3, y3) = filter.apply(20.0, 30.0);
    assert!((x3 - 14.05).abs() < 0.001);
    assert!((y3 - 24.05).abs() < 0.001);
}

#[test]
fn test_median_python_parity() {
    let mut filter = create_filter("median:3").unwrap();
    
    for (i, &(x_in, y_in)) in python_reference::TEST_SEQUENCE.iter().enumerate() {
        let (x_out, y_out) = filter.apply(x_in, y_in);
        let (x_expected, y_expected) = python_reference::MEDIAN_3_EXPECTED[i];
        
        assert!(
            (x_out - x_expected).abs() < TOLERANCE,
            "Median X mismatch at index {}: got {}, expected {}",
            i, x_out, x_expected
        );
        assert!(
            (y_out - y_expected).abs() < TOLERANCE,
            "Median Y mismatch at index {}: got {}, expected {}",
            i, y_out, y_expected
        );
    }
}

#[test]
fn test_lowpass_filter_behavior() {
    // Test with known alpha value
    let mut filter = create_filter("lowpass:0.5115").unwrap();
    
    // First value should pass through
    let (x1, y1) = filter.apply(10.0, 20.0);
    assert_eq!((x1, y1), (10.0, 20.0));
    
    // Second value with alpha ~0.5115
    let (x2, y2) = filter.apply(15.0, 25.0);
    let expected_x = 0.5115 * 15.0 + (1.0 - 0.5115) * 10.0;
    let expected_y = 0.5115 * 25.0 + (1.0 - 0.5115) * 20.0;
    assert!((x2 - expected_x).abs() < 0.01);
    assert!((y2 - expected_y).abs() < 0.01);
}

/// Test with outlier data to verify Hampel filter behavior
#[test]
fn test_hampel_outlier_rejection() {
    let mut filter = create_filter("hampel:5:3.0").unwrap();
    
    // Feed normal values
    let normal_values = vec![
        (10.0, 20.0),
        (11.0, 21.0),
        (12.0, 22.0),
        (13.0, 23.0),
    ];
    
    for &(x, y) in &normal_values {
        let _ = filter.apply(x, y);
    }
    
    // Feed an outlier
    let (x_out, y_out) = filter.apply(100.0, 200.0);
    
    // The outlier should be rejected (should be close to median of previous values)
    assert!(
        x_out < 20.0,
        "Hampel filter should reject X outlier, got {}",
        x_out
    );
    assert!(
        y_out < 40.0,
        "Hampel filter should reject Y outlier, got {}",
        y_out
    );
}

/// Test filter reset functionality matches Python behavior
#[test]
fn test_filter_reset_python_parity() {
    let mut filter = create_filter("exponential:0.5").unwrap();
    
    // Apply some values
    filter.apply(100.0, 200.0);
    filter.apply(150.0, 250.0);
    
    // Reset filter
    filter.reset();
    
    // First value after reset should initialize the filter
    let (x, y) = filter.apply(50.0, 60.0);
    assert_eq!((x, y), (50.0, 60.0), "First value after reset should pass through");
    
    // Second value should be filtered
    let (x2, y2) = filter.apply(70.0, 80.0);
    assert_eq!((x2, y2), (60.0, 70.0), "0.5 * 50 + 0.5 * 70 = 60");
}

/// Test edge case: single value in window
#[test]
fn test_single_value_edge_case() {
    // Moving average with single value
    {
        let mut filter = create_filter("movingaverage:5").unwrap();
        let (x, y) = filter.apply(42.0, 84.0);
        assert_eq!((x, y), (42.0, 84.0), "Single value should pass through");
    }
    
    // Median with single value
    {
        let mut filter = create_filter("median:5").unwrap();
        let (x, y) = filter.apply(42.0, 84.0);
        assert_eq!((x, y), (42.0, 84.0), "Single value should pass through");
    }
}

/// Test numerical stability with extreme values
#[test]
fn test_numerical_stability() {
    let extreme_values = vec![
        (1e6, -1e6),    // Large values
        (1e-6, -1e-6),  // Small values
        (0.0, 0.0),     // Zero
    ];
    
    let filter_types = vec!["movingaverage:3", "exponential:0.5", "median:3", "lowpass:0.5"];
    
    for filter_type in filter_types {
        let mut filter = create_filter(filter_type).unwrap();
        
        for &(x, y) in &extreme_values {
            let (x_out, y_out) = filter.apply(x, y);
            assert!(
                x_out.is_finite() && y_out.is_finite(),
                "{} produced non-finite output for ({}, {})",
                filter_type, x, y
            );
        }
    }
}