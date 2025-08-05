//! Error types for the head pose estimation library.

use thiserror::Error;

/// Main error type for the library
#[derive(Error, Debug)]
pub enum Error {
    /// `OpenCV` operation failed
    #[error("OpenCV error: {0}")]
    OpenCV(#[from] opencv::Error),

    /// `ONNX` Runtime inference failed
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(#[from] ort::OrtError),

    /// File I/O operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image processing operation failed
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    /// `X11` window system operation failed
    #[error("X11 error: {0}")]
    X11(String),

    /// Invalid input parameters provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Model loading or inference error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Filter initialization or processing error
    #[error("Filter error: {0}")]
    FilterError(String),

    /// Movement detection algorithm error
    #[error("Movement detection error: {0}")]
    MovementDetectionError(String),
    
    /// Cursor control operation failed
    #[error("Cursor control error: {0}")]
    CursorControl(String),
}

/// Application-specific error type (alias for main Error type)
pub type AppError = Error;

/// Convenience type alias for Results with our Error type
pub type Result<T> = std::result::Result<T, Error>;
