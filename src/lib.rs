//! Head pose estimation library for real-time human head pose tracking.
//!
//! This library provides a Rust implementation of head pose estimation using:
//! - ONNX Runtime for deep learning inference
//! - `OpenCV` for computer vision operations
//! - Multiple filtering algorithms for smoothing pose estimates
//!
//! The estimation pipeline consists of:
//! 1. Face detection to locate faces in the image
//! 2. Facial landmark detection to find 68 key points
//! 3. Pose estimation using `PnP` (Perspective-n-Point) algorithm
//! 4. Optional filtering to smooth the results
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```no_run
//! use head_pose_estimation::{face_detection::FaceDetector, mark_detection::MarkDetector,
//!                            pose_estimation::PoseEstimator, filters::create_filter};
//! use opencv::{imgcodecs, core::Mat, prelude::*};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load models
//! let mut face_detector = FaceDetector::new("assets/face_detector.onnx", 0.5, 0.5)?;
//! let mark_detector = MarkDetector::new("assets/face_landmarks.onnx")?;
//! let pose_estimator = PoseEstimator::new("assets/model.txt", 640, 480)?;
//!
//! // Load and process an image
//! let image = imgcodecs::imread("test.jpg", imgcodecs::IMREAD_COLOR)?;
//!
//! // Detect faces
//! let faces = face_detector.detect(&image)?;
//!
//! // For each detected face
//! for face in faces {
//!     // Extract face region
//!     let face_roi = Mat::roi(&image, face.bbox)?;
//!     
//!     // Detect landmarks
//!     let face_roi_mat = face_roi.try_clone()?;
//!     let landmarks = mark_detector.detect(&face_roi_mat)?;
//!     
//!     // Estimate pose
//!     // Convert landmarks to tuple format
//!     let landmark_tuples: Vec<(f32, f32)> = landmarks.iter()
//!         .map(|p| (p.x, p.y))
//!         .collect();
//!     let (euler_angles, _, _) = pose_estimator.estimate_pose(&landmark_tuples)?;
//!     let pitch = euler_angles[0];
//!     let yaw = euler_angles[1];
//!     println!("Pitch: {:.2}°, Yaw: {:.2}°", pitch, yaw);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Filters
//!
//! ```no_run
//! use head_pose_estimation::filters::{CursorFilter, create_filter};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a Kalman filter
//! let mut filter = create_filter("kalman")?;
//!
//! // Apply filtering to pose estimates
//! // Assume we have pitch and yaw values from pose estimation
//! let pitch = 10.5;
//! let yaw = -15.2;
//!
//! let filtered = filter.apply(pitch, yaw);
//! println!("Filtered - Pitch: {:.2}°, Yaw: {:.2}°", filtered.0, filtered.1);
//!
//! // Reset filter if needed
//! filter.reset();
//! # Ok(())
//! # }
//! ```
//!
//! ## Movement Detection
//!
//! ```no_run
//! use head_pose_estimation::movement_detector::MovementDetector;
//!
//! # fn main() {
//! // Create movement detector
//! let mut detector = MovementDetector::new(30, 0.5);
//!
//! // Update with pose estimates
//! let pitch = 10.5;
//! let yaw = -15.2;
//! let is_moving = detector.update(pitch, yaw);
//!
//! if is_moving {
//!     println!("Head is moving");
//! } else {
//!     println!("Head is still");
//! }
//!
//! // Get movement statistics
//! if let Some((pitch_stats, yaw_stats)) = detector.get_stats() {
//!     println!("Pitch std dev: {:.2}", pitch_stats.std_dev);
//!     println!("Yaw std dev: {:.2}", yaw_stats.std_dev);
//! }
//! # }
//! ```
//!
//! ## Complete Pipeline Example
//!
//! ```no_run
//! use head_pose_estimation::{
//!     face_detection::FaceDetector,
//!     mark_detection::MarkDetector,
//!     pose_estimation::PoseEstimator,
//!     filters::{CursorFilter, create_filter},
//!     movement_detector::MovementDetector,
//! };
//! use opencv::{videoio, core::Mat, imgproc, highgui, prelude::*};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize components
//! let mut face_detector = FaceDetector::new("assets/face_detector.onnx", 0.5, 0.5)?;
//! let mark_detector = MarkDetector::new("assets/face_landmarks.onnx")?;
//! let pose_estimator = PoseEstimator::new("assets/model.txt", 640, 480)?;
//! let mut filter = create_filter("kalman")?;
//! let mut movement_detector = MovementDetector::new(30, 0.5);
//!
//! // Open webcam
//! let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
//! let mut frame = Mat::default();
//!
//! loop {
//!     // Read frame
//!     if !cap.read(&mut frame)? {
//!         break;
//!     }
//!
//!     // Detect faces
//!     let faces = face_detector.detect(&frame)?;
//!     
//!     for face in faces {
//!         // Extract face region
//!         let face_roi = Mat::roi(&frame, face.bbox)?;
//!         
//!         // Detect landmarks
//!         let face_roi_mat = face_roi.try_clone()?;
//!         let landmarks = mark_detector.detect(&face_roi_mat)?;
//!         
//!         // Estimate pose
//!         // Convert landmarks to tuple format
//!         let landmark_tuples: Vec<(f32, f32)> = landmarks.iter()
//!             .map(|p| (p.x, p.y))
//!             .collect();
//!         let (euler_angles, _, _) = pose_estimator.estimate_pose(&landmark_tuples)?;
//!         let pitch = euler_angles[0];
//!         let yaw = euler_angles[1];
//!         
//!         // Apply filtering
//!         let (filtered_pitch, filtered_yaw) = filter.apply(pitch, yaw);
//!         
//!         // Detect movement
//!         let is_moving = movement_detector.update(filtered_pitch, filtered_yaw);
//!         
//!         // Draw bounding box
//!         imgproc::rectangle(
//!             &mut frame,
//!             face.bbox,
//!             opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
//!             2,
//!             imgproc::LINE_8,
//!             0,
//!         )?;
//!     }
//!     
//!     // Display frame
//!     highgui::imshow("Head Pose Estimation", &frame)?;
//!     
//!     // Exit on 'q' key
//!     if highgui::wait_key(1)? == b'q' as i32 {
//!         break;
//!     }
//! }
//! # Ok(())
//! # }
//! ```

/// Face detection module for finding faces in images
pub mod face_detection;

/// Facial landmark detection module for finding 68 key points
pub mod mark_detection;

/// Head pose estimation module using `PnP` algorithm
pub mod pose_estimation;

/// Signal filtering algorithms for smoothing pose estimates
pub mod filters;

/// Movement detection module for tracking head movement patterns
pub mod movement_detector;

/// Utility functions for image processing and coordinate transformations
pub mod utils;

/// Error types and result handling
pub mod error;

/// Main application module
pub mod app;

/// Cursor control module for X11 systems
pub mod cursor_control;

/// Constants used throughout the application
pub mod constants;

/// Configuration management
pub mod config;

pub use error::{Error, Result};
