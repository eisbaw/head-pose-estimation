//! Tests for video input processing with various formats and content types

use std::process::Command;
use std::fs;

/// Generate a test video with colored frames using ffmpeg
fn generate_test_video(output_path: &str, duration_seconds: u32, fps: u32, resolution: &str) -> Result<(), String> {
    // Create a temporary directory for test videos
    fs::create_dir_all("test_videos").map_err(|e| format!("Failed to create test_videos dir: {}", e))?;
    
    let output = Command::new("ffmpeg")
        .args([
            "-y", // Overwrite output file
            "-f", "lavfi", // Use libavfilter input
            "-i", &format!("testsrc=duration={}:size={}:rate={}", duration_seconds, resolution, fps),
            "-vf", "format=yuv420p", // Ensure compatibility
            "-c:v", "libx264", // Use H.264 codec
            "-preset", "ultrafast", // Fast encoding
            output_path,
        ])
        .output()
        .map_err(|e| format!("Failed to execute ffmpeg: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg failed: {}", stderr));
    }
    
    Ok(())
}

/// Generate a test video with a face (using a simple pattern that might trigger face detection)
fn generate_face_test_video(output_path: &str) -> Result<(), String> {
    // Create a video with patterns that might be detected as faces
    // This uses a test pattern with circular shapes
    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "lavfi",
            "-i", "testsrc2=duration=3:size=640x480:rate=30",
            "-vf", "format=yuv420p,drawbox=x=250:y=150:w=140:h=180:color=white:t=fill,\
                    drawbox=x=280:y=190:w=20:h=20:color=black:t=fill,\
                    drawbox=x=340:y=190:w=20:h=20:color=black:t=fill,\
                    drawbox=x=310:y=250:w=20:h=40:color=black:t=fill",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            output_path,
        ])
        .output()
        .map_err(|e| format!("Failed to execute ffmpeg: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg failed: {}", stderr));
    }
    
    Ok(())
}

#[test]
#[ignore = "Requires ffmpeg and ONNX models"]
fn test_video_format_mp4() {
    let video_path = "test_videos/test_format.mp4";
    
    // Generate test video
    if let Err(e) = generate_test_video(video_path, 2, 30, "640x480") {
        eprintln!("Skipping test: {}", e);
        return;
    }
    
    // Run head pose estimation on the video
    let output = Command::new("cargo")
        .args([
            "run", "--release", "--",
            "--video", video_path,
            "--gui", "none",
            "--filter", "kalman",
        ])
        .output()
        .expect("Failed to execute head-pose-estimation");
    
    // The program should run without crashing
    // It may fail due to no faces detected, but shouldn't crash
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panic") && !stderr.contains("SIGSEGV"),
        "Program crashed: {}",
        stderr
    );
    
    // Clean up
    let _ = fs::remove_file(video_path);
}

#[test]
#[ignore = "Requires ffmpeg and ONNX models"]
fn test_different_resolutions() {
    let resolutions = vec![
        ("320x240", "test_videos/test_320x240.mp4"),
        ("640x480", "test_videos/test_640x480.mp4"),
        ("1280x720", "test_videos/test_1280x720.mp4"),
        ("1920x1080", "test_videos/test_1920x1080.mp4"),
    ];
    
    for (resolution, video_path) in resolutions {
        println!("Testing resolution: {}", resolution);
        
        // Generate test video
        if let Err(e) = generate_test_video(video_path, 1, 30, resolution) {
            eprintln!("Skipping resolution {}: {}", resolution, e);
            continue;
        }
        
        // Run head pose estimation
        let output = Command::new("cargo")
            .args([
                "run", "--release", "--",
                "--video", video_path,
                "--gui", "none",
                "--filter", "none",
            ])
            .output()
            .expect("Failed to execute head-pose-estimation");
        
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panic") && !stderr.contains("SIGSEGV"),
            "Program crashed with resolution {}: {}",
            resolution,
            stderr
        );
        
        // Clean up
        let _ = fs::remove_file(video_path);
    }
}

#[test]
#[ignore = "Requires ffmpeg and ONNX models"]
fn test_different_framerates() {
    let framerates = vec![15, 24, 30, 60];
    
    for fps in framerates {
        println!("Testing framerate: {} fps", fps);
        
        let video_path = format!("test_videos/test_{}fps.mp4", fps);
        
        // Generate test video
        if let Err(e) = generate_test_video(&video_path, 2, fps, "640x480") {
            eprintln!("Skipping framerate {}: {}", fps, e);
            continue;
        }
        
        // Run head pose estimation
        let output = Command::new("cargo")
            .args([
                "run", "--release", "--",
                "--video", &video_path,
                "--gui", "none",
                "--filter", "median",
            ])
            .output()
            .expect("Failed to execute head-pose-estimation");
        
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panic") && !stderr.contains("SIGSEGV"),
            "Program crashed with framerate {}: {}",
            fps,
            stderr
        );
        
        // Clean up
        let _ = fs::remove_file(&video_path);
    }
}

#[test]
#[ignore = "Requires ffmpeg and ONNX models"]
fn test_video_with_face_pattern() {
    let video_path = "test_videos/test_face_pattern.mp4";
    
    // Generate test video with face-like pattern
    if let Err(e) = generate_face_test_video(video_path) {
        eprintln!("Skipping test: {}", e);
        return;
    }
    
    // Run with all filters to test processing
    let filters = vec!["none", "kalman", "median", "exponential", "lowpass"];
    
    for filter in filters {
        println!("Testing with filter: {}", filter);
        
        let output = Command::new("cargo")
            .args([
                "run", "--release", "--",
                "--video", video_path,
                "--gui", "none",
                "--filter", filter,
            ])
            .output()
            .expect("Failed to execute head-pose-estimation");
        
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panic") && !stderr.contains("SIGSEGV"),
            "Program crashed with filter {}: {}",
            filter,
            stderr
        );
    }
    
    // Clean up
    let _ = fs::remove_file(video_path);
}

#[test]
#[ignore = "Requires ffmpeg and ONNX models"]
fn test_short_video_duration() {
    // Test very short videos (potential edge case)
    let durations = vec![1, 5, 10]; // seconds
    
    for duration in durations {
        println!("Testing duration: {} seconds", duration);
        
        let video_path = format!("test_videos/test_{}s.mp4", duration);
        
        if let Err(e) = generate_test_video(&video_path, duration, 30, "640x480") {
            eprintln!("Skipping duration {}: {}", duration, e);
            continue;
        }
        
        let output = Command::new("cargo")
            .args([
                "run", "--release", "--",
                "--video", &video_path,
                "--gui", "none",
            ])
            .output()
            .expect("Failed to execute head-pose-estimation");
        
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panic") && !stderr.contains("SIGSEGV"),
            "Program crashed with duration {}: {}",
            duration,
            stderr
        );
        
        // Clean up
        let _ = fs::remove_file(&video_path);
    }
}

#[test]
fn test_video_file_not_found() {
    let output = Command::new("cargo")
        .args([
            "run", "--",
            "--video", "nonexistent_video.mp4",
            "--gui", "none",
        ])
        .output()
        .expect("Failed to execute head-pose-estimation");
    
    assert!(!output.status.success(), "Should fail with nonexistent video");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to open video file") || stderr.contains("ModelError"),
        "Should report video file error"
    );
}

/// Clean up test videos directory after all tests
#[test]
#[ignore]
fn cleanup_test_videos() {
    let _ = fs::remove_dir_all("test_videos");
}