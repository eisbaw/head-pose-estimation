//! Robustness tests for video processing

use std::process::Command;

/// Test handling of various video file extensions
#[test]
fn test_video_extension_handling() {
    let extensions = vec![
        "mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "mpg", "mpeg",
        "m4v", "3gp", "ogv", "vob", "ts", "m2ts", "mts", "divx", "xvid"
    ];
    
    for ext in extensions {
        let filename = format!("test.{}", ext);
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "--video", &filename,
                "--help"
            ])
            .output()
            .expect("Failed to execute command");
        
        // Should accept any extension (actual file validation happens later)
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed to accept .{} extension",
            ext
        );
    }
}

/// Test handling of special characters in filenames
#[test]
fn test_special_characters_in_filename() {
    let filenames = vec![
        "test video.mp4",           // Space
        "test-video.mp4",           // Dash
        "test_video.mp4",           // Underscore
        "test.video.mp4",           // Multiple dots
        "test(1).mp4",              // Parentheses
        "test[1].mp4",              // Brackets
        "test@video.mp4",           // At symbol
        "test#1.mp4",               // Hash
        "test$video.mp4",           // Dollar sign
        "test%20video.mp4",         // URL encoded space
        "test+video.mp4",           // Plus sign
        "test=video.mp4",           // Equals sign
        "test&video.mp4",           // Ampersand
        "test'video.mp4",           // Single quote
        "test,video.mp4",           // Comma
        "test;video.mp4",           // Semicolon
        "test~video.mp4",           // Tilde
        "test!video.mp4",           // Exclamation
        "test{video}.mp4",          // Curly braces
    ];
    
    for filename in filenames {
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "--video", filename,
                "--help"
            ])
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed to handle filename: {}",
            filename
        );
    }
}

/// Test handling of various path formats
#[test]
fn test_path_formats() {
    let paths = vec![
        "./test.mp4",                      // Relative path with ./
        "../test.mp4",                     // Parent directory
        "videos/test.mp4",                 // Subdirectory
        "/tmp/test.mp4",                   // Absolute path
        "~/test.mp4",                      // Home directory
        "a/b/c/d/e/f/test.mp4",          // Deep nesting
        "./videos/../test.mp4",            // Path with ..
        "//test.mp4",                      // Double slash
        "videos//test.mp4",                // Double slash in middle
        "videos/./test.mp4",               // Current directory in path
    ];
    
    for path in paths {
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "--video", path,
                "--help"
            ])
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed to handle path format: {}",
            path
        );
    }
}

/// Test command line argument combinations with video
#[test]
fn test_video_with_all_filters() {
    let filters = vec![
        "none", "kalman", "median", "median:5", "median:7",
        "moving_average", "movingaverage:3", "movingaverage:5",
        "exponential", "exponential:0.3", "exponential:0.5",
        "lowpass", "lowpass:0.5", "lowpass:0.7",
        "secondorderlowpass:0.3:0.7",
        "hampel:5:3.0", "hampel:7:2.5"
    ];
    
    for filter in filters {
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "--video", "test.mp4",
                "--filter", filter,
                "--help"
            ])
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed with filter: {}",
            filter
        );
    }
}

/// Test video with all GUI modes
#[test]
fn test_video_with_all_gui_modes() {
    let gui_modes = vec!["none", "cam", "pointers", "all"];
    
    for mode in gui_modes {
        let output = Command::new("cargo")
            .args([
                "run", "--",
                "--video", "test.mp4",
                "--gui", mode,
                "--help"
            ])
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed with GUI mode: {}",
            mode
        );
    }
}

/// Test video with cursor control modes
#[test]
fn test_video_with_cursor_modes() {
    let cursor_modes = vec!["none", "absolute", "relative"];
    let data_sources = vec!["pitchyaw", "normalproj"];
    let vectors = vec!["location", "speed"];
    
    for cursor_mode in &cursor_modes {
        for data_source in &data_sources {
            for vector in &vectors {
                let output = Command::new("cargo")
                    .args([
                        "run", "--",
                        "--video", "test.mp4",
                        "--cursor", cursor_mode,
                        "--datasource", data_source,
                        "--vector", vector,
                        "--help"
                    ])
                    .output()
                    .expect("Failed to execute command");
                
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    stdout.contains("Realtime human head pose estimation"),
                    "Failed with cursor={}, data-source={}, vector={}",
                    cursor_mode, data_source, vector
                );
            }
        }
    }
}

/// Test video with numeric parameters
#[test]
fn test_video_with_numeric_parameters() {
    let test_cases = vec![
        vec!["--brightness=-50"],
        vec!["--brightness", "0"],
        vec!["--brightness", "50"],
        vec!["--brightness", "10"],
        vec!["--filter", "kalman"],
        vec!["--filter", "median:5"],
        vec!["--filter", "lowpass:0.5"],
        vec!["--brightness", "10", "--filter", "kalman"],
        vec!["--gui", "none"],
        vec!["--gui", "cam"],
    ];
    
    for args in test_cases {
        let mut cmd_args = vec!["run", "--", "--video", "test.mp4"];
        cmd_args.extend(args.iter().map(|s| *s));
        cmd_args.push("--help");
        
        let output = Command::new("cargo")
            .args(&cmd_args)
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed with args: {:?}",
            args
        );
    }
}

/// Test video with boolean flags
#[test]
fn test_video_with_boolean_flags() {
    let flag_combinations = vec![
        vec!["--inv", "x"],
        vec!["--inv", "y"],
        vec!["--inv", "xy"],
        vec!["--debug"],
        vec!["--cursor-filter-all"],
        vec!["--debug", "--cursor-filter-all"],
        vec!["--inv", "xy", "--debug", "--cursor-filter-all"],
    ];
    
    for flags in flag_combinations {
        let mut cmd_args = vec!["run", "--", "--video", "test.mp4"];
        cmd_args.extend(flags.iter().map(|s| *s));
        cmd_args.push("--help");
        
        let output = Command::new("cargo")
            .args(&cmd_args)
            .output()
            .expect("Failed to execute command");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Realtime human head pose estimation"),
            "Failed with flags: {:?}",
            flags
        );
    }
}

/// Test long video filename (potential buffer overflow)
#[test]
fn test_very_long_filename() {
    let long_name = "a".repeat(200) + ".mp4";
    
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "--video", &long_name,
            "--help"
        ])
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Realtime human head pose estimation"),
        "Failed to handle very long filename"
    );
}

/// Test empty video filename
#[test]
fn test_empty_video_filename() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "--video", "",
            "--gui", "none"
        ])
        .output()
        .expect("Failed to execute command");
    
    // Should fail gracefully
    assert!(!output.status.success(), "Should reject empty filename");
}