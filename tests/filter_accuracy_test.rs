//! Tests for filter output accuracy comparing with expected values

use head_pose_estimation::filters::create_filter;

/// Test that filters produce expected output values
#[test]
fn test_filter_output_accuracy() {
    // Test moving average filter
    {
        let mut filter = create_filter("movingaverage:3").unwrap();
        
        // Feed values and check output
        let (x1, y1) = filter.apply(3.0, 6.0);
        assert_eq!((x1, y1), (3.0, 6.0), "First value should pass through");
        
        let (x2, y2) = filter.apply(6.0, 9.0);
        assert_eq!((x2, y2), (4.5, 7.5), "Average of [3,6] and [6,9]");
        
        let (x3, y3) = filter.apply(9.0, 12.0);
        assert_eq!((x3, y3), (6.0, 9.0), "Average of [3,6], [6,9], [9,12]");
        
        let (x4, y4) = filter.apply(12.0, 15.0);
        assert_eq!((x4, y4), (9.0, 12.0), "Average of [6,9], [9,12], [12,15]");
    }
    
    // Test median filter
    {
        let mut filter = create_filter("median:3").unwrap();
        
        let _ = filter.apply(1.0, 1.0);
        let _ = filter.apply(5.0, 5.0);
        let (x3, y3) = filter.apply(3.0, 3.0);
        assert_eq!((x3, y3), (3.0, 3.0), "Median of [1,5,3] is 3");
        
        let (x4, y4) = filter.apply(7.0, 7.0);
        assert_eq!((x4, y4), (5.0, 5.0), "Median of [5,3,7] is 5");
    }
    
    // Test exponential filter
    {
        let mut filter = create_filter("exponential:0.5").unwrap();
        
        let (x1, y1) = filter.apply(10.0, 20.0);
        assert_eq!((x1, y1), (10.0, 20.0), "First value initializes filter");
        
        let (x2, y2) = filter.apply(20.0, 40.0);
        assert_eq!((x2, y2), (15.0, 30.0), "0.5 * 10 + 0.5 * 20 = 15");
        
        let (x3, y3) = filter.apply(30.0, 60.0);
        assert_eq!((x3, y3), (22.5, 45.0), "0.5 * 15 + 0.5 * 30 = 22.5");
    }
    
    // Test no filter (passthrough)
    {
        let mut filter = create_filter("none").unwrap();
        
        let test_values = vec![
            (1.23, 4.56),
            (-10.0, 20.0),
            (0.0, 0.0),
            (100.0, -100.0),
        ];
        
        for &(x, y) in &test_values {
            let (out_x, out_y) = filter.apply(x, y);
            assert_eq!((out_x, out_y), (x, y), "NoFilter should pass values through unchanged");
        }
    }
}

/// Test filter convergence behavior
#[test]
fn test_filter_convergence_accuracy() {
    // Test that exponential filter converges to constant input
    {
        let mut filter = create_filter("exponential:0.1").unwrap();
        
        // Apply constant value many times
        for _ in 0..100 {
            let _ = filter.apply(42.0, 84.0);
        }
        
        let (x, y) = filter.apply(42.0, 84.0);
        assert!((x - 42.0).abs() < 0.001, "Should converge to 42.0, got {}", x);
        assert!((y - 84.0).abs() < 0.001, "Should converge to 84.0, got {}", y);
    }
    
    // Test that moving average converges to constant input
    {
        let mut filter = create_filter("movingaverage:5").unwrap();
        
        // Fill buffer with constant values
        for _ in 0..10 {
            let _ = filter.apply(33.0, 66.0);
        }
        
        let (x, y) = filter.apply(33.0, 66.0);
        assert_eq!((x, y), (33.0, 66.0), "Moving average of constant values should equal the constant");
    }
}

/// Test filter impulse response accuracy
#[test]
fn test_filter_impulse_response_accuracy() {
    // Test low-pass filter response to step input
    {
        let mut filter = create_filter("lowpass:0.1").unwrap();
        
        // Initialize with zero
        let _ = filter.apply(0.0, 0.0);
        
        // Apply step input
        let mut responses = Vec::new();
        for _ in 0..10 {
            let (x, _) = filter.apply(100.0, 0.0);
            responses.push(x);
        }
        
        // Check that response is monotonically increasing
        for i in 1..responses.len() {
            assert!(responses[i] > responses[i-1], "Low-pass filter should monotonically approach step input");
        }
        
        // Check specific values (exponential response)
        assert!((responses[0] - 10.0).abs() < 0.001, "First response should be alpha * input");
        assert!(responses[9] > 65.0, "Should be significantly closer to target after 10 steps");
    }
}

/// Test Hampel filter outlier rejection accuracy
#[test]
fn test_hampel_filter_accuracy() {
    let mut filter = create_filter("hampel:7:3.0").unwrap();
    
    // Feed normal values
    for i in 0..6 {
        let val = i as f64 * 10.0;
        let _ = filter.apply(val, val);
    }
    
    // Feed an outlier
    let (x_outlier, y_outlier) = filter.apply(1000.0, 1000.0);
    
    // The outlier should be rejected and replaced with median
    assert!(x_outlier < 100.0, "Hampel filter should reject outlier, got {}", x_outlier);
    assert!(y_outlier < 100.0, "Hampel filter should reject outlier, got {}", y_outlier);
    
    // Continue with normal values
    let (x_normal, y_normal) = filter.apply(70.0, 70.0);
    assert!((x_normal - 70.0).abs() < 20.0, "Should accept normal value close to trend");
    assert!((y_normal - 70.0).abs() < 20.0, "Should accept normal value close to trend");
}

/// Test second-order low-pass filter accuracy
#[test]
fn test_second_order_lowpass_accuracy() {
    // Use higher cutoff frequency for faster convergence
    let mut filter = create_filter("secondorderlowpass:0.3:0.7").unwrap();
    
    // Test step response
    let _ = filter.apply(0.0, 0.0);
    
    let mut x_vals = Vec::new();
    for _ in 0..50 {
        let (x, _) = filter.apply(100.0, 0.0);
        x_vals.push(x);
    }
    
    // Should have damped response (no overshoot with damping = 0.7)
    let max_val = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(max_val <= 100.0, "Should not overshoot target with damping = 0.7");
    
    // Should converge close to target
    assert!(x_vals[49] > 85.0, "Should converge close to target value, got {}", x_vals[49]);
    
    // Also test that it's increasing monotonically (no oscillation)
    for i in 5..49 {
        assert!(x_vals[i+1] >= x_vals[i] - 0.001, "Should increase monotonically with damping = 0.7");
    }
}

/// Test Kalman filter accuracy with noisy data
#[test]
fn test_kalman_filter_accuracy() {
    let mut filter = create_filter("kalman").unwrap();
    
    // Simulate noisy measurements of a constant value
    let true_value = 50.0;
    let noise_amplitude = 5.0;
    
    let mut filtered_values = Vec::new();
    for i in 0..50 {
        // Add deterministic "noise"
        let noise = ((i * 7) % 11) as f64 - 5.0; // Range -5 to 5
        let noisy_value = true_value + noise * noise_amplitude / 5.0;
        
        let (x, _) = filter.apply(noisy_value, noisy_value);
        if i >= 10 { // Let filter stabilize
            filtered_values.push(x);
        }
    }
    
    // Calculate mean of filtered values
    let mean = filtered_values.iter().sum::<f64>() / filtered_values.len() as f64;
    
    // Kalman filter should reduce noise and get close to true value
    assert!((mean - true_value).abs() < 2.0, "Kalman filter mean {} should be close to true value {}", mean, true_value);
    
    // Calculate variance
    let variance = filtered_values.iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>() / filtered_values.len() as f64;
    
    // Variance should be low (filter reduces noise)
    assert!(variance < 5.0, "Kalman filter should reduce variance, got {}", variance);
}