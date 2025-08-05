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

pub use error::{Error, Result};
