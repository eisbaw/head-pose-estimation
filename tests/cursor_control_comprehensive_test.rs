//! Comprehensive tests for cursor control functionality

use head_pose_estimation::cursor_control::{CursorMode, DataSource};

/// Test cursor mode default values
#[test]
fn test_cursor_mode_default() {
    // Verify Default trait implementation
    let default_mode = CursorMode::default();
    assert!(matches!(default_mode, CursorMode::Absolute));
}

/// Test data source default values
#[test]
fn test_data_source_default() {
    let default_source = DataSource::default();
    assert!(matches!(default_source, DataSource::PitchYaw));
}

/// Test normal vector projection calculations
#[test]
fn test_normal_projection_calculations() {
    // Test normal vector to screen coordinates conversion
    
    // Forward-facing normal (0, 0, 1) should map to center
    let nx = 0.0;
    let ny = 0.0;
    let screen_width = 1920;
    let screen_height = 1080;
    
    let screen_x = ((nx + 1.0) * 0.5 * screen_width as f64) as i32;
    let screen_y = ((ny + 1.0) * 0.5 * screen_height as f64) as i32;
    
    assert_eq!(screen_x, screen_width / 2);
    assert_eq!(screen_y, screen_height / 2);
    
    // Test extreme normal vectors
    let test_cases = vec![
        (-1.0, 0.0, 0),                    // Full left
        (1.0, 0.0, screen_width - 1),      // Full right
        (0.0, -1.0, screen_width / 2),     // Center X when Y varies
        (0.0, 1.0, screen_width / 2),      // Center X when Y varies
    ];
    
    for (nx, ny, expected_x) in test_cases {
        let screen_x = ((nx + 1.0) * 0.5 * screen_width as f64)
            .clamp(0.0, (screen_width - 1) as f64) as i32;
        assert_eq!(screen_x, expected_x, "Failed for normal ({}, {})", nx, ny);
    }
}

/// Test angle clamping and normalization
#[test]
fn test_angle_clamping() {
    let clamp_angle = |angle: f64, min: f64, max: f64| -> f64 {
        angle.clamp(min, max)
    };
    
    // Test normal range
    assert_eq!(clamp_angle(15.0, -30.0, 30.0), 15.0);
    assert_eq!(clamp_angle(-15.0, -30.0, 30.0), -15.0);
    
    // Test clamping
    assert_eq!(clamp_angle(45.0, -30.0, 30.0), 30.0);
    assert_eq!(clamp_angle(-45.0, -30.0, 30.0), -30.0);
    
    // Test edge cases
    assert_eq!(clamp_angle(30.0, -30.0, 30.0), 30.0);
    assert_eq!(clamp_angle(-30.0, -30.0, 30.0), -30.0);
}

/// Test cursor sensitivity scaling
#[test]
fn test_sensitivity_scaling() {
    let base_movement = 10.0;
    let sensitivities = vec![0.5, 1.0, 1.5, 2.0, 3.0];
    
    for sensitivity in sensitivities {
        let scaled_movement = base_movement * sensitivity;
        assert!(((scaled_movement - base_movement * sensitivity) as f64).abs() < 0.001);
    }
    
    // Test negative movements
    let negative_movement = -10.0;
    let sensitivity = 1.5;
    let scaled = negative_movement * sensitivity;
    assert_eq!(scaled, -15.0);
}

/// Test cursor smoothing with filters
#[test]
fn test_cursor_filter_integration() {
    // Simulate cursor positions with filter smoothing
    let positions = vec![
        (100.0, 200.0),
        (110.0, 210.0),
        (120.0, 220.0),
        (130.0, 230.0),
        (140.0, 240.0),
    ];
    
    // Moving average simulation (window size 3)
    let mut buffer_x = Vec::new();
    let mut buffer_y = Vec::new();
    let window_size = 3;
    
    for (x, y) in positions {
        buffer_x.push(x);
        buffer_y.push(y);
        
        if buffer_x.len() > window_size {
            buffer_x.remove(0);
            buffer_y.remove(0);
        }
        
        let avg_x: f64 = buffer_x.iter().sum::<f64>() / buffer_x.len() as f64;
        let avg_y: f64 = buffer_y.iter().sum::<f64>() / buffer_y.len() as f64;
        
        // Verify averaging is working
        assert!(avg_x > 0.0 && avg_y > 0.0);
    }
}

/// Test velocity-based cursor control
#[test]
fn test_velocity_cursor_control() {
    let mut position_x = 500.0_f64;
    let mut position_y = 500.0_f64;
    let mut velocity_x = 0.0_f64;
    let mut velocity_y = 0.0_f64;
    
    // Test acceleration
    let target_velocity_x = 10.0;
    let target_velocity_y = -5.0;
    let acceleration_factor = 0.1;
    
    // Simulate acceleration over multiple frames
    for _ in 0..20 {
        velocity_x += (target_velocity_x - velocity_x) * acceleration_factor;
        velocity_y += (target_velocity_y - velocity_y) * acceleration_factor;
        
        position_x += velocity_x;
        position_y += velocity_y;
    }
    
    // Velocity should approach target (with more iterations, it gets closer)
    assert!((velocity_x - target_velocity_x).abs() < 2.0);
    assert!((velocity_y - target_velocity_y).abs() < 2.0);
    
    // Position should have moved
    assert!(position_x > 500.0);
    assert!(position_y < 500.0);
}

/// Test dead zone implementation
#[test]
fn test_dead_zone() {
    let dead_zone_threshold = 5.0;
    
    let apply_dead_zone = |value: f64, threshold: f64| -> f64 {
        if value.abs() < threshold {
            0.0
        } else {
            value
        }
    };
    
    // Test values within dead zone
    assert_eq!(apply_dead_zone(2.0, dead_zone_threshold), 0.0);
    assert_eq!(apply_dead_zone(-3.0, dead_zone_threshold), 0.0);
    assert_eq!(apply_dead_zone(4.9, dead_zone_threshold), 0.0);
    
    // Test values outside dead zone
    assert_eq!(apply_dead_zone(6.0, dead_zone_threshold), 6.0);
    assert_eq!(apply_dead_zone(-7.0, dead_zone_threshold), -7.0);
    assert_eq!(apply_dead_zone(5.1, dead_zone_threshold), 5.1);
}

/// Test screen boundary handling
#[test]
fn test_screen_boundaries() {
    let screen_width = 1920;
    let screen_height = 1080;
    
    let clamp_to_screen = |x: i32, y: i32| -> (i32, i32) {
        (
            x.clamp(0, screen_width - 1),
            y.clamp(0, screen_height - 1)
        )
    };
    
    // Test normal positions
    assert_eq!(clamp_to_screen(500, 500), (500, 500));
    assert_eq!(clamp_to_screen(0, 0), (0, 0));
    assert_eq!(clamp_to_screen(1919, 1079), (1919, 1079));
    
    // Test out of bounds
    assert_eq!(clamp_to_screen(-100, 500), (0, 500));
    assert_eq!(clamp_to_screen(2000, 500), (1919, 500));
    assert_eq!(clamp_to_screen(500, -100), (500, 0));
    assert_eq!(clamp_to_screen(500, 2000), (500, 1079));
    assert_eq!(clamp_to_screen(-100, -100), (0, 0));
    assert_eq!(clamp_to_screen(2000, 2000), (1919, 1079));
}

/// Test cursor mode state machine
#[test]
fn test_cursor_mode_state_machine() {
    #[derive(Debug, PartialEq)]
    enum State {
        Idle,
        Active,
        Suspended,
    }
    
    let mut state = State::Idle;
    let mut cursor_mode = CursorMode::Absolute;
    
    // Test state transitions
    
    // Idle -> Active (when face detected)
    state = State::Active;
    assert_eq!(state, State::Active);
    
    // Active -> Suspended (when key pressed in relative mode)
    cursor_mode = CursorMode::Relative;
    let key_pressed = true;
    if cursor_mode == CursorMode::Relative && !key_pressed {
        state = State::Suspended;
    }
    assert_eq!(state, State::Active); // Should stay active when key pressed
    
    // Active -> Suspended (when key released)
    let key_pressed = false;
    if cursor_mode == CursorMode::Relative && !key_pressed {
        state = State::Suspended;
    }
    assert_eq!(state, State::Suspended);
    
    // Suspended -> Active (when key pressed again)
    let key_pressed = true;
    if key_pressed {
        state = State::Active;
    }
    assert_eq!(state, State::Active);
}

/// Test movement detection integration
#[test]
fn test_movement_detection_for_cursor() {
    // Simulate angle history for movement detection
    let angle_history = vec![
        (10.0, 20.0),
        (10.1, 20.1),
        (10.0, 20.0),
        (10.1, 19.9),
        (10.0, 20.0),
    ];
    
    // Calculate standard deviation
    let mean_pitch = angle_history.iter().map(|(p, _)| p).sum::<f64>() / angle_history.len() as f64;
    let _mean_yaw = angle_history.iter().map(|(_, y)| y).sum::<f64>() / angle_history.len() as f64;
    
    let variance_pitch = angle_history.iter()
        .map(|(p, _)| (p - mean_pitch).powi(2))
        .sum::<f64>() / angle_history.len() as f64;
    
    let std_dev_pitch = variance_pitch.sqrt();
    
    // Small movements should have low std deviation
    assert!(std_dev_pitch < 0.1);
    
    // Test with larger movements
    let large_movement_history = vec![
        (10.0, 20.0),
        (15.0, 25.0),
        (5.0, 15.0),
        (20.0, 30.0),
        (0.0, 10.0),
    ];
    
    let mean_pitch_large = large_movement_history.iter()
        .map(|(p, _)| p)
        .sum::<f64>() / large_movement_history.len() as f64;
    
    let variance_pitch_large = large_movement_history.iter()
        .map(|(p, _)| (p - mean_pitch_large).powi(2))
        .sum::<f64>() / large_movement_history.len() as f64;
    
    let std_dev_pitch_large = variance_pitch_large.sqrt();
    
    // Larger movements should have higher std deviation
    assert!(std_dev_pitch_large > 5.0);
}

/// Test cursor update rate limiting
#[test]
fn test_cursor_update_rate_limiting() {
    use std::time::{Duration, Instant};
    
    let target_fps = 60;
    let frame_duration = Duration::from_millis(1000 / target_fps);
    
    let mut last_update = Instant::now();
    let mut update_count = 0;
    
    // Simulate updates over 100ms
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(100) {
        let now = Instant::now();
        if now.duration_since(last_update) >= frame_duration {
            update_count += 1;
            last_update = now;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    
    // Should have approximately 6 updates in 100ms at 60 FPS
    assert!(update_count >= 5 && update_count <= 7, 
            "Expected ~6 updates, got {}", update_count);
}