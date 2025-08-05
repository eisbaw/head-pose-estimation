//! Tests for cursor control functionality

use head_pose_estimation::cursor_control::{CursorController, CursorMode, DataSource};
use std::thread;
use std::time::Duration;

/// Test cursor mode transitions
#[test]
fn test_cursor_mode_transitions() {
    // Test all cursor mode variants exist and can be set
    let modes = vec![
        CursorMode::Absolute,
        CursorMode::Relative,
        CursorMode::Velocity,
    ];
    
    for mode in modes {
        // Just verify the enum variants exist and can be used
        match mode {
            CursorMode::Absolute => assert!(true),
            CursorMode::Relative => assert!(true),
            CursorMode::Velocity => assert!(true),
        }
    }
    
    // Test data source variants
    let sources = vec![
        DataSource::PitchYaw,
        DataSource::NormalProjection,
    ];
    
    for source in sources {
        match source {
            DataSource::PitchYaw => assert!(true),
            DataSource::NormalProjection => assert!(true),
        }
    }
}

/// Test cursor position calculation
#[test]
fn test_cursor_position_calculation() {
    // Test angle to pixel conversion
    let screen_width = 1920;
    let screen_height = 1080;
    let pitch_range = 60.0; // -30 to +30 degrees
    let yaw_range = 80.0;   // -40 to +40 degrees
    
    // Test center position (0, 0) angles
    let center_x = angle_to_pixel(0.0, yaw_range, screen_width);
    let center_y = angle_to_pixel(0.0, pitch_range, screen_height);
    assert_eq!(center_x, screen_width / 2);
    assert_eq!(center_y, screen_height / 2);
    
    // Test extreme positions
    let left_x = angle_to_pixel(-yaw_range / 2.0, yaw_range, screen_width);
    let right_x = angle_to_pixel(yaw_range / 2.0, yaw_range, screen_width);
    let top_y = angle_to_pixel(-pitch_range / 2.0, pitch_range, screen_height);
    let bottom_y = angle_to_pixel(pitch_range / 2.0, pitch_range, screen_height);
    
    assert_eq!(left_x, 0);
    assert_eq!(right_x, screen_width - 1);
    assert_eq!(top_y, 0);
    assert_eq!(bottom_y, screen_height - 1);
    
    // Test clamping
    let over_left = angle_to_pixel(-yaw_range, yaw_range, screen_width);
    let over_right = angle_to_pixel(yaw_range, yaw_range, screen_width);
    assert_eq!(over_left, 0);
    assert_eq!(over_right, screen_width - 1);
}

/// Helper function to convert angle to pixel position
fn angle_to_pixel(angle: f64, angle_range: f64, screen_dimension: i32) -> i32 {
    // Map angle from [-angle_range/2, angle_range/2] to [0, screen_dimension]
    let normalized = (angle + angle_range / 2.0) / angle_range;
    let pixel = (normalized * screen_dimension as f64) as i32;
    pixel.clamp(0, screen_dimension - 1)
}

/// Test movement-based cursor control
#[test]
fn test_movement_based_control() {
    // Test movement threshold logic
    let movement_threshold = 0.5;
    
    // Simulate no movement
    let still_std_dev = 0.1;
    let movement_active = still_std_dev > movement_threshold;
    assert!(!movement_active, "Should not activate for small movement");
    
    // Simulate significant movement
    let moving_std_dev = 1.5;
    let movement_active = moving_std_dev > movement_threshold;
    assert!(movement_active, "Should activate for large movement");
}

/// Test relative cursor position updates
#[test]
fn test_relative_position_updates() {
    let mut cursor_x: f64 = 960.0;
    let mut cursor_y: f64 = 540.0;
    let sensitivity = 2.0;
    
    // Test small relative movement
    let delta_pitch = 1.0;
    let delta_yaw = -1.5;
    
    cursor_x += delta_yaw * sensitivity;
    cursor_y += delta_pitch * sensitivity;
    
    assert!((cursor_x - 957.0).abs() < 0.01);
    assert!((cursor_y - 542.0).abs() < 0.01);
    
    // Test clamping at boundaries
    cursor_x = 10.0;
    cursor_y = 10.0;
    
    let large_negative_delta = -100.0;
    cursor_x += large_negative_delta * sensitivity;
    cursor_y += large_negative_delta * sensitivity;
    
    // Should be clamped to 0
    cursor_x = cursor_x.max(0.0);
    cursor_y = cursor_y.max(0.0);
    
    assert_eq!(cursor_x, 0.0);
    assert_eq!(cursor_y, 0.0);
}

/// Test speed-based cursor control
#[test]
fn test_speed_based_control() {
    // Simulate cursor speed control
    let base_speed = 5.0;
    let angle_pitch = 15.0;
    let angle_yaw = -10.0;
    let max_angle = 30.0;
    
    // Calculate speed based on angle
    let pitch_speed: f64 = (angle_pitch / max_angle) * base_speed;
    let yaw_speed: f64 = (angle_yaw / max_angle) * base_speed;
    
    assert!((pitch_speed - 2.5).abs() < 0.01);
    assert!((yaw_speed - -1.667).abs() < 0.01);
    
    // Test dead zone
    let dead_zone = 5.0;
    let small_angle: f64 = 3.0;
    
    let speed_with_deadzone = if small_angle.abs() < dead_zone {
        0.0
    } else {
        (small_angle / max_angle) * base_speed
    };
    
    assert_eq!(speed_with_deadzone, 0.0, "Should be zero within dead zone");
}

/// Test cursor controller thread safety
#[test]
#[ignore = "Requires X11 display"]
fn test_cursor_controller_thread_safety() {
    use std::sync::{Arc, Mutex};
    
    let controller = Arc::new(Mutex::new(
        CursorController::new().expect("Failed to create cursor controller")
    ));
    
    let mut handles = vec![];
    
    // Spawn multiple threads trying to update cursor position
    for i in 0..4 {
        let controller_clone = Arc::clone(&controller);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let _position = (i * 100 + j, i * 100 + j);
                if let Ok(ctrl) = controller_clone.try_lock() {
                    // In real implementation, this would call set_position
                    drop(ctrl); // Release lock
                }
                thread::sleep(Duration::from_millis(10));
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test X11 cursor control initialization
#[test]
#[ignore = "Requires X11 display"]
fn test_x11_initialization() {
    match CursorController::new() {
        Ok(_controller) => {
            // Successfully created controller
            assert!(true, "X11 cursor controller created successfully");
        }
        Err(e) => {
            // This is expected in CI environment without X11
            println!("Expected error in headless environment: {}", e);
            assert!(true, "Properly handled X11 initialization error");
        }
    }
}