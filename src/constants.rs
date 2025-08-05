//! Constants used throughout the application

/// Number of facial landmarks for full face
pub const NUM_FACIAL_LANDMARKS: usize = 68;

/// Default frames per second assumption
pub const DEFAULT_FPS: f64 = 30.0;

/// Total number of 3D model coordinates (68 points × 3 dimensions)
pub const MODEL_POINTS_TOTAL_VALUES: usize = 204;

/// Image normalization constants for face detection
pub const IMAGE_NORMALIZATION_OFFSET: f32 = 127.5;
pub const IMAGE_NORMALIZATION_SCALE: f32 = 128.0;

/// Default brightness adjustment value
pub const DEFAULT_BRIGHTNESS: i32 = 0;

/// Angular conversion constants
pub const OMEGA_FACTOR: f64 = 2.0;

/// Camera matrix center factor
pub const CAMERA_CENTER_FACTOR: f64 = 2.0;

/// Default window sizes for filters
pub const DEFAULT_MOVING_AVERAGE_WINDOW: usize = 5;
pub const DEFAULT_MEDIAN_WINDOW: usize = 5;
pub const DEFAULT_HAMPEL_WINDOW: usize = 5;

/// Default filter parameters
pub const DEFAULT_EXPONENTIAL_ALPHA: f64 = 0.5;
pub const DEFAULT_LOW_PASS_CUTOFF: f64 = 0.5;
pub const DEFAULT_SECOND_ORDER_CUTOFF: f64 = 30.0;
pub const DEFAULT_SECOND_ORDER_DAMPING: f64 = 0.707;
pub const DEFAULT_HAMPEL_THRESHOLD: f64 = 3.0;

/// Movement detection default threshold
pub const DEFAULT_MOVEMENT_THRESHOLD: f64 = 2.0;

/// Statistical constants
pub const SQRT_2: f64 = 1.414_213_562_373_095_1;

/// Exponential filter bounds
pub const EXPONENTIAL_ALPHA_MIN: f64 = 0.0;
pub const EXPONENTIAL_ALPHA_MAX: f64 = 1.0;

/// Numeric precision epsilon
pub const EPSILON: f64 = 1e-10;