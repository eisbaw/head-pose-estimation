//! Configuration management for the head pose estimation application

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Model configuration
    pub models: ModelConfig,

    /// Face detection configuration
    pub face_detection: FaceDetectionConfig,

    /// Filter configuration
    pub filter: FilterConfig,

    /// Display configuration
    pub display: DisplayConfig,

    /// Cursor control configuration
    pub cursor: CursorConfig,

    /// Movement detection configuration
    pub movement: MovementConfig,
}

/// Model file paths configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to face detection ONNX model
    pub face_detector: PathBuf,

    /// Path to facial landmarks ONNX model
    pub face_landmarks: PathBuf,

    /// Path to 3D face model points
    pub face_model_3d: PathBuf,
}

/// Face detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetectionConfig {
    /// Confidence threshold for face detection (0.0-1.0)
    pub confidence_threshold: f32,

    /// IOU threshold for non-maximum suppression (0.0-1.0)
    pub iou_threshold: f32,

    /// Maximum number of faces to detect
    pub max_faces: usize,

    /// Face region expansion factor
    pub bbox_expansion: f32,
}

/// Filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Default filter type
    pub default_filter: String,

    /// Moving average window size
    pub moving_average_window: usize,

    /// Median filter window size
    pub median_window: usize,

    /// Exponential filter alpha value
    pub exponential_alpha: f64,

    /// Low pass filter alpha
    pub low_pass_alpha: f64,

    /// Second order low pass cutoff frequency
    pub second_order_cutoff: f64,

    /// Second order low pass damping
    pub second_order_damping: f64,

    /// Hampel filter window size
    pub hampel_window: usize,

    /// Hampel filter threshold
    pub hampel_threshold: f64,
}

/// Display configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Target framerate
    pub target_fps: u32,

    /// Window width
    pub window_width: i32,

    /// Window height
    pub window_height: i32,

    /// Default GUI mode
    pub gui_mode: String,

    /// Show filter comparison
    pub show_filters: bool,

    /// Flip image horizontally
    pub flip_x: bool,

    /// Flip image vertically
    pub flip_y: bool,

    /// Brightness adjustment (-100 to 100)
    pub brightness: i32,
}

/// Cursor control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorConfig {
    /// Enable cursor control
    pub enabled: bool,

    /// Invert X axis
    pub invert_x: bool,

    /// Invert Y axis
    pub invert_y: bool,

    /// X axis amplification
    pub amplify_x: f64,

    /// Y axis amplification
    pub amplify_y: f64,

    /// Use normal vector projection
    pub use_normal_vector: bool,

    /// Default cursor mode
    pub default_mode: String,
}

/// Movement detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementConfig {
    /// Window size for statistics calculation
    pub window_size: usize,

    /// Standard deviation threshold for movement detection
    pub std_dev_threshold: f64,

    /// Enable movement-based cursor control
    pub movement_based_control: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models: ModelConfig::default(),
            face_detection: FaceDetectionConfig::default(),
            filter: FilterConfig::default(),
            display: DisplayConfig::default(),
            cursor: CursorConfig::default(),
            movement: MovementConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            face_detector: PathBuf::from("assets/face_detector.onnx"),
            face_landmarks: PathBuf::from("assets/face_landmarks.onnx"),
            face_model_3d: PathBuf::from("assets/model.txt"),
        }
    }
}

impl Default for FaceDetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            iou_threshold: 0.5,
            max_faces: 10,
            bbox_expansion: 0.25,
        }
    }
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            default_filter: "moving_average".to_string(),
            moving_average_window: 5,
            median_window: 5,
            exponential_alpha: 0.5,
            low_pass_alpha: 0.3,
            second_order_cutoff: 10.0,
            second_order_damping: 0.7,
            hampel_window: 7,
            hampel_threshold: 3.0,
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            target_fps: 30,
            window_width: 640,
            window_height: 480,
            gui_mode: "all".to_string(),
            show_filters: false,
            flip_x: false,
            flip_y: false,
            brightness: 0,
        }
    }
}

impl Default for CursorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            invert_x: false,
            invert_y: false,
            amplify_x: 3.0,
            amplify_y: 3.0,
            use_normal_vector: false,
            default_mode: "absolute".to_string(),
        }
    }
}

impl Default for MovementConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            std_dev_threshold: 0.5,
            movement_based_control: false,
        }
    }
}

impl Config {
    /// Load configuration from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::IoError(e.to_string()))?;

        serde_yaml::from_str(&content).map_err(|e| Error::ConfigError(format!("Failed to parse config: {}", e)))
    }

    /// Save configuration to a YAML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| Error::ConfigError(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content).map_err(|e| Error::IoError(e.to_string()))?;

        Ok(())
    }

    /// Create a filter from configuration
    pub fn create_filter(&self) -> Result<Box<dyn crate::filters::CursorFilter>> {
        use crate::filters::{
            create_filter,
            exponential::ExponentialFilter,
            hampel::HampelFilter,
            low_pass::{LowPassFilter, SecondOrderLowPassFilter},
            median::MedianFilter,
            moving_average::MovingAverageFilter,
        };

        match self.filter.default_filter.as_str() {
            "moving_average" => Ok(Box::new(MovingAverageFilter::new(self.filter.moving_average_window))),
            "median" => Ok(Box::new(MedianFilter::new(self.filter.median_window))),
            "exponential" => Ok(Box::new(ExponentialFilter::new(self.filter.exponential_alpha))),
            "low_pass" => Ok(Box::new(LowPassFilter::new(self.filter.low_pass_alpha))),
            "second_order_low_pass" => Ok(Box::new(SecondOrderLowPassFilter::new(
                self.filter.second_order_cutoff,
                self.filter.second_order_damping,
            ))),
            "hampel" => Ok(Box::new(HampelFilter::new(
                self.filter.hampel_window,
                self.filter.hampel_threshold,
            ))),
            name => create_filter(name),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate thresholds
        if !(0.0..=1.0).contains(&self.face_detection.confidence_threshold) {
            return Err(Error::ConfigError(
                "Confidence threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.face_detection.iou_threshold) {
            return Err(Error::ConfigError(
                "IOU threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate filter parameters
        if self.filter.moving_average_window == 0 {
            return Err(Error::ConfigError(
                "Moving average window size must be greater than 0".to_string(),
            ));
        }
        if self.filter.median_window == 0 || self.filter.median_window % 2 == 0 {
            return Err(Error::ConfigError(
                "Median window size must be odd and greater than 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.filter.exponential_alpha) {
            return Err(Error::ConfigError(
                "Exponential alpha must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate display settings
        if self.display.target_fps == 0 {
            return Err(Error::ConfigError("Target FPS must be greater than 0".to_string()));
        }
        if !(-100..=100).contains(&self.display.brightness) {
            return Err(Error::ConfigError(
                "Brightness must be between -100 and 100".to_string(),
            ));
        }

        // Validate model paths exist
        if !self.models.face_detector.exists() {
            return Err(Error::ConfigError(format!(
                "Face detector model not found: {}",
                self.models.face_detector.display()
            )));
        }
        if !self.models.face_landmarks.exists() {
            return Err(Error::ConfigError(format!(
                "Face landmarks model not found: {}",
                self.models.face_landmarks.display()
            )));
        }
        if !self.models.face_model_3d.exists() {
            return Err(Error::ConfigError(format!(
                "3D face model not found: {}",
                self.models.face_model_3d.display()
            )));
        }

        Ok(())
    }
}

/// Example configuration file content
pub const EXAMPLE_CONFIG: &str = r#"# Head Pose Estimation Configuration

# Model paths
models:
  face_detector: "assets/face_detector.onnx"
  face_landmarks: "assets/face_landmarks.onnx"
  face_model_3d: "assets/model.txt"

# Face detection parameters
face_detection:
  confidence_threshold: 0.6
  iou_threshold: 0.5
  max_faces: 10
  bbox_expansion: 0.25

# Filter configuration
filter:
  default_filter: "moving_average"
  moving_average_window: 5
  median_window: 5
  exponential_alpha: 0.5
  low_pass_alpha: 0.3
  second_order_cutoff: 10.0
  second_order_damping: 0.7
  hampel_window: 7
  hampel_threshold: 3.0

# Display settings
display:
  target_fps: 30
  window_width: 640
  window_height: 480
  gui_mode: "all"
  show_filters: false
  flip_x: false
  flip_y: false
  brightness: 0

# Cursor control
cursor:
  enabled: true
  invert_x: false
  invert_y: false
  amplify_x: 3.0
  amplify_y: 3.0
  use_normal_vector: false
  default_mode: "absolute"

# Movement detection
movement:
  window_size: 30
  std_dev_threshold: 0.5
  movement_based_control: false
"#;
