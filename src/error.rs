use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("OpenCV error: {0}")]
    OpenCV(#[from] opencv::Error),
    
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(#[from] ort::OrtError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),
    
    #[error("X11 error: {0}")]
    X11(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Filter error: {0}")]
    FilterError(String),
    
    #[error("Movement detection error: {0}")]
    MovementDetectionError(String),
}

pub type Result<T> = std::result::Result<T, Error>;