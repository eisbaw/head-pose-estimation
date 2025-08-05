//! Edge case tests for filters, movement detection, and other components

use head_pose_estimation::{filters::create_filter, movement_detector::MovementDetector};

#[test]
fn test_filter_extreme_values() {
    let filters = vec![
        "none",
        "movingaverage:5",
        "median:5",
        "exponential:0.8",
        "kalman",
        "lowpass:0.1",
        "secondorderlowpass:0.1:1.0",
        "hampel:7:3.0",
    ];

    for filter_str in filters {
        let mut filter = create_filter(filter_str).unwrap();

        // Test with extreme values
        let extreme_values = vec![
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY),
            (f64::NAN, f64::NAN),
            (f64::MAX, f64::MIN),
            (1e100, -1e100),
            (0.0, 0.0),
        ];

        for (x, y) in extreme_values {
            let (out_x, out_y) = filter.apply(x, y);

            // For most filters, NaN and infinity should propagate or be handled gracefully
            // We just ensure the filter doesn't panic
            // Some filters like Kalman might produce infinity from extreme values
            let _ = (out_x, out_y);
        }
    }
}

#[test]
fn test_filter_reset_behavior() {
    let filters = vec![
        "movingaverage:3",
        "median:3",
        "exponential:0.5",
        "kalman",
        "lowpass:0.5",
        "secondorderlowpass:0.5:0.7",
        "hampel:5:2.5",
    ];

    for filter_str in filters {
        let mut filter = create_filter(filter_str).unwrap();

        // Apply some values
        filter.apply(10.0, 20.0);
        filter.apply(15.0, 25.0);
        filter.apply(20.0, 30.0);

        // Store the output
        let before_reset = filter.apply(25.0, 35.0);

        // Reset
        filter.reset();

        // The same input should give different output after reset
        let after_reset = filter.apply(25.0, 35.0);

        // For most filters, the output should be different after reset
        // NoFilter and some robust filters might be exceptions
        if filter.name() != "NoFilter" && filter.name() != "HampelFilter" && filter.name() != "MedianFilter" {
            // These filters might give the same output on the first sample after reset
            assert!(
                (before_reset.0 - after_reset.0).abs() > 1e-10 || (before_reset.1 - after_reset.1).abs() > 1e-10,
                "Filter {} did not change output after reset",
                filter.name()
            );
        }
    }
}

#[test]
fn test_movement_detector_edge_cases() {
    // Test with window size of 1
    let mut detector = MovementDetector::new(1, 1.0);
    let moving = detector.update(10.0, 20.0);
    assert!(!moving, "Single sample should not trigger movement");

    // Test with very high threshold
    let mut detector = MovementDetector::new(5, f64::MAX);
    for _ in 0..10 {
        let moving = detector.update(rand::random::<f64>() * 1000.0, rand::random::<f64>() * 1000.0);
        assert!(!moving, "Infinite threshold should never trigger movement");
    }

    // Test with zero threshold
    let mut detector = MovementDetector::new(5, 0.0);
    detector.update(10.0, 20.0);
    detector.update(10.0, 20.0);
    detector.update(10.0, 20.0);
    detector.update(10.0, 20.0);
    let moving = detector.update(10.0000001, 20.0000001);
    assert!(moving, "Zero threshold should trigger on tiny changes");
}

#[test]
fn test_movement_detector_reset() {
    let mut detector = MovementDetector::new(5, 2.0);

    // Fill with movement data
    for i in 0..5 {
        detector.update(i as f64 * 10.0, i as f64 * 10.0);
    }

    let stats_before = detector.get_stats();
    assert!(stats_before.is_some());

    // Reset
    detector.reset();

    // Stats should be None after reset
    let stats_after = detector.get_stats();
    assert!(stats_after.is_none());

    // Buffer should be empty
    assert!(!detector.update(0.0, 0.0)); // First update after reset shouldn't detect movement
}

#[test]
fn test_filter_convergence() {
    // Test that filters converge to steady state with constant input
    let filters = vec![
        "movingaverage:10",
        "exponential:0.8",
        "kalman",
        "lowpass:0.1",
        "secondorderlowpass:0.1:0.7",
    ];

    for filter_str in filters {
        let mut filter = create_filter(filter_str).unwrap();
        let target = (42.0, 84.0);

        // Apply the same value many times
        let mut last_output = (0.0, 0.0);
        for _ in 0..100 {
            last_output = filter.apply(target.0, target.1);
        }

        // Should converge close to the target
        assert!(
            (last_output.0 - target.0).abs() < 1.0,
            "Filter {} did not converge to target x value",
            filter.name()
        );
        assert!(
            (last_output.1 - target.1).abs() < 1.0,
            "Filter {} did not converge to target y value",
            filter.name()
        );
    }
}

#[test]
fn test_filter_impulse_response() {
    // Test how filters respond to impulses
    let filters = vec!["movingaverage:5", "median:5", "exponential:0.8", "hampel:5:3.0"];

    for filter_str in filters {
        let mut filter = create_filter(filter_str).unwrap();

        // Establish baseline
        for _ in 0..10 {
            filter.apply(10.0, 10.0);
        }

        // Apply impulse
        let impulse_response = filter.apply(1000.0, 1000.0);

        // Return to baseline
        let recovery_response = filter.apply(10.0, 10.0);

        match filter.name() {
            "MedianFilter" | "HampelFilter" => {
                // These should reject the impulse
                assert!(impulse_response.0 < 500.0, "{} did not reject impulse", filter.name());
            }
            _ => {
                // Others will be affected by the impulse
                assert!(
                    impulse_response.0 > 50.0,
                    "{} was not affected by impulse",
                    filter.name()
                );
            }
        }

        // All should start recovering (or at least not get worse)
        // Some filters like median might immediately recover
        assert!(recovery_response.0 <= impulse_response.0);
    }
}

// Note: Using a simple RNG for test determinism
mod rand {
    use std::cell::RefCell;

    thread_local! {
        static SEED: RefCell<u64> = RefCell::new(12345);
    }

    #[allow(unused)]
    pub fn random<T>() -> f64 {
        SEED.with(|seed| {
            let mut s = seed.borrow_mut();
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((*s / 65536) % 32768) as f64 / 32768.0
        })
    }
}
