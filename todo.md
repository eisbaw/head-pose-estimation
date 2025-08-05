# Head Pose Estimation Rust Port - Task List

## Phase 1: Project Setup and Infrastructure

### Week 1: Basic Setup
- [x] DONE: Create new Rust project with Cargo
- [ ] Set up workspace structure with multiple crates if needed
- [x] DONE: Add initial dependencies to Cargo.toml
- [x] DONE: Create module structure matching Python architecture
- [x] DONE: Set up error handling with anyhow/thiserror
- [x] DONE: Configure logging with env_logger
- [ ] Create build script for system dependency detection
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [x] DONE: Add rustfmt.toml and clippy configuration (in Cargo.toml)

### Week 2: Core Infrastructure
- [ ] Implement basic image type conversions (OpenCV Mat <-> ndarray)
- [x] DONE: Create bounding box struct and operations (using opencv::core::Rect)
- [x] DONE: Port `refine()` function from utils.py
- [ ] Implement basic geometry helpers
- [x] DONE: Create trait for cursor filters
- [ ] Set up ONNX Runtime initialization
- [ ] Add configuration struct for application settings
- [x] DONE: Implement command-line argument parsing with clap

## Phase 2: ONNX Model Integration

### Week 3: Face Detection Module
- [x] DONE: Create FaceDetector struct
- [ ] Implement ONNX model loading for face_detector.onnx
- [ ] Port image preprocessing for face detection
- [ ] Implement forward pass inference
- [x] DONE: Port distance2bbox function
- [ ] Port distance2kps function
- [ ] Implement NMS (Non-Maximum Suppression)
- [ ] Add face detection visualization
- [x] DONE: Write unit tests for face detection (test_distance_to_bbox)

### Week 4: Landmark Detection Module
- [x] DONE: Create MarkDetector struct
- [ ] Implement ONNX model loading for face_landmarks.onnx
- [ ] Port image preprocessing (resize, color conversion)
- [ ] Implement batch inference support
- [ ] Convert landmark output format
- [ ] Add landmark visualization
- [ ] Write unit tests for landmark detection

## Phase 3: Pose Estimation

### Week 5: Pose Estimation Core
- [x] DONE: Create PoseEstimator struct
- [x] DONE: Load 3D model points from assets/model.txt
- [x] DONE: Implement camera matrix initialization
- [x] DONE: Port PnP solver using OpenCV
- [x] DONE: Implement Euler angle extraction
- [ ] Port pose visualization (3D box drawing)
- [ ] Implement draw_axes functionality
- [ ] Implement draw_normal_vector functionality
- [x] DONE: Add tests for pose estimation accuracy (test_euler_angle_conversion)

## Phase 4: Filtering System

### Week 6: Basic Filters
- [x] DONE: Define CursorFilter trait
- [x] DONE: Implement NoFilter
- [x] DONE: Implement MovingAverageFilter
- [x] DONE: Implement MedianFilter
- [x] DONE: Implement ExponentialFilter
- [x] DONE: Create filter factory function
- [x] DONE: Add filter reset functionality
- [x] DONE: Write unit tests for each filter

### Week 7: Advanced Filters
- [x] DONE: Implement KalmanFilter with nalgebra
- [x] DONE: Implement LowPassFilter (1st order)
- [x] DONE: Implement SecondOrderLowPassFilter
- [x] DONE: Implement HampelFilter
- [ ] Optimize filter performance
- [ ] Add benchmarks for filters
- [ ] Ensure numerical accuracy matches Python

## Phase 5: Application Features

### Week 8: Movement Detection and Utilities
- [x] DONE: Create MovementDetector struct
- [x] DONE: Implement sliding window buffer
- [x] DONE: Port statistical calculations
- [x] DONE: Add movement detection logic
- [x] DONE: Implement debug statistics
- [ ] Create utility functions for screen resolution
- [ ] Add X11 cursor control functions
- [ ] Implement keyboard state detection

### Week 9: Video Processing Pipeline
- [ ] Set up OpenCV VideoCapture
- [ ] Implement frame reading loop
- [ ] Add webcam buffer optimization
- [ ] Implement image flipping/inversion options
- [ ] Add brightness adjustment
- [ ] Create frame timing/FPS counter
- [ ] Implement graceful shutdown
- [ ] Add video file support

## Phase 6: Main Application

### Week 10: Core Application Logic
- [ ] Implement main application struct
- [ ] Port argument parsing logic
- [ ] Create GUI window management
- [ ] Implement cursor position mapping (angles to pixels)
- [ ] Implement normal vector projection mode
- [ ] Add multiple filter visualization mode
- [ ] Implement GUI mode selection (all/pointers/cam/none)
- [ ] Add legend and UI text rendering

### Week 11: Cursor Control Features
- [ ] Implement absolute cursor control mode
- [ ] Implement relative cursor control mode
- [ ] Add 'w' key detection for relative mode
- [ ] Implement location vector mode
- [ ] Implement speed vector mode
- [ ] Create cursor update thread for speed mode
- [ ] Add movement-based cursor control
- [ ] Test cursor control on X11

## Phase 7: Testing and Optimization

### Week 12: Integration and Testing
- [ ] Create integration tests for full pipeline
- [ ] Add performance benchmarks vs Python
- [ ] Profile memory usage
- [ ] Optimize hot paths
- [ ] Add SIMD optimizations where beneficial
- [ ] Test with various video inputs
- [ ] Ensure filter outputs match Python within tolerance
- [ ] Create test suite for cursor control

### Week 13: Polish and Documentation
- [ ] Write comprehensive README
- [ ] Add inline documentation
- [ ] Create usage examples
- [ ] Document build instructions
- [ ] Add troubleshooting guide
- [ ] Create migration guide from Python
- [ ] Package for distribution
- [ ] Create release binaries

## Phase 8: Future Enhancements (Optional)

### Platform Support
- [ ] Abstract cursor control for Wayland support
- [ ] Add Windows support
- [ ] Add macOS support
- [ ] Create platform-specific builds

### Performance Optimizations
- [ ] Implement GPU acceleration for inference
- [ ] Add multi-face tracking support
- [ ] Optimize filter calculations with SIMD
- [ ] Implement frame skipping for low-end hardware

### Additional Features
- [ ] Add configuration file support
- [ ] Implement filter parameter tuning UI
- [ ] Add recording/playback functionality
- [ ] Create headless mode for servers
- [ ] Add REST API for remote control

## Testing Checklist

### Unit Tests
- [ ] Test each filter algorithm
- [ ] Test pose estimation math
- [ ] Test movement detection logic
- [ ] Test coordinate transformations
- [ ] Test ONNX model loading

### Integration Tests
- [ ] Test full face detection pipeline
- [ ] Test landmark detection accuracy
- [ ] Test pose estimation accuracy
- [ ] Test cursor control modes
- [ ] Test video file processing

### Performance Tests
- [ ] Benchmark vs Python implementation
- [ ] Measure memory usage
- [ ] Test real-time performance (30+ FPS)
- [ ] Profile CPU usage
- [ ] Test with different video resolutions

### Compatibility Tests
- [ ] Test with original ONNX models
- [ ] Test all command-line arguments
- [ ] Test filter output accuracy
- [ ] Test on different Linux distributions
- [ ] Test with different OpenCV versions

## Dependencies to Add

```toml
[dependencies]
opencv = { version = "0.88", features = ["opencv-4", "contrib"] }
ort = { version = "1.16", features = ["download-binaries"] }
nalgebra = "0.32"
clap = { version = "4.4", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
x11rb = "0.12"
rayon = "1.7"
log = "0.4"
env_logger = "0.10"
ndarray = "0.15"
image = "0.24"
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

## Build Requirements

- [ ] Set up shell.nix with Rust toolchain
- [ ] Add ONNX Runtime to shell.nix
- [ ] Configure OpenCV in shell.nix
- [ ] Create justfile with build commands
- [ ] Add development dependencies
- [ ] Set up pre-commit hooks
- [ ] Configure rustfmt and clippy

## Success Metrics

- [ ] All unit tests passing
- [ ] Feature parity with Python version
- [ ] Performance equal or better than Python
- [ ] Memory usage lower than Python
- [ ] Single binary under 50MB
- [ ] Startup time under 1 second
- [ ] Maintains 30+ FPS on reference hardware

## Immediate Tasks (Added by MPED)

### Code Quality
- [x] DONE: Fix unsafe code warnings in pose_estimation.rs
- [x] DONE: Fix dead code warnings (unused struct fields)
- [x] DONE: Fix unused imports and variables warnings
- [x] DONE: Add missing documentation for all public items
- [x] DONE: Add comprehensive module-level documentation
- [x] DONE: Document error types and their meanings

### Implementation TODOs
- [x] DONE: Implement ONNX model loading in FaceDetector::new
- [x] DONE: Implement face detection preprocessing in FaceDetector::detect
- [x] DONE: Implement face detection inference
- [x] DONE: Implement face detection postprocessing with NMS
- [x] DONE: Implement ONNX model loading in MarkDetector::new
- [x] DONE: Implement landmark detection preprocessing
- [x] DONE: Implement landmark detection inference
- [x] DONE: Port distance2kps function for face detection

### Missing Core Functionality
- [x] DONE: Implement main application loop
- [x] DONE: Implement video capture from webcam
- [x] DONE: Implement video capture from file
- [x] DONE: Implement cursor control (X11)
- [x] DONE: Implement GUI visualization
- [x] DONE: Implement filter mode switching (show all filters for comparison)
- [x] DONE: Implement debug overlay (angles, movement status, statistics)

### Testing
- [x] DONE: Add integration tests for full pipeline
- [x] DONE: Add performance benchmarks
- [x] DONE: Test with actual ONNX models
- [x] DONE: Test cursor control functionality

## Code Quality Issues (Found by Clippy)

### High Priority
- [x] DONE: Fix potential panic in KalmanFilter matrix inversion (already uses try_inverse)
- [x] DONE: Fix suspicious operation grouping in SecondOrderLowPassFilter (line 96)
- [ ] Add proper error handling for all unwrap() calls in tests
- [ ] Handle usize to i32 casts that may wrap on 32-bit systems
- [ ] Add more backticks to documentation items (OpenCV, PnP, window_size)
- [x] DONE: Handle partial_cmp unwrap calls in sort operations

### Medium Priority
- [x] DONE: Fix clippy lint group priorities in Cargo.toml
- [x] DONE: Add #[must_use] to constructors that return Self
- [x] DONE: Use f64::from() instead of as f64 for lossless casts
- [x] DONE: Use .copied() instead of .cloned() for f64 values
- [x] DONE: Use mul_add for better floating point accuracy
- [x] DONE: Add # Panics sections to functions that can panic

### Low Priority
- [x] DONE: Make movement_detector::calculate_stats an associated function
- [x] DONE: Add backticks to documentation (PnP, refine())
- [x] DONE: Use format! with inline variables
- [ ] Consider making placeholder detect() functions const
- [x] DONE: Remove redundant import in main.rs
- [x] DONE: Consider removing Result return from main()

## Missing Tests

### Unit Tests
- [x] DONE: Test refine_boxes edge cases (empty boxes, negative values)
- [x] DONE: Test filter reset functionality for all filters
- [ ] Test MovementDetector with edge cases
- [ ] Test PoseEstimator::parse_model_points with invalid input
- [ ] Test error handling in all modules

### Integration Tests
- [ ] Test filter chain with multiple filters
- [ ] Test pose estimation with real landmark data
- [ ] Test movement detection with real pose data

## New Code Quality Issues Found

### Performance Considerations
- [ ] Consider using SmallVec for small collections in filters
- [ ] Profile VecDeque vs Vec performance for fixed-size buffers
- [ ] Investigate SIMD opportunities in filter calculations
- [ ] Consider zero-copy optimizations for OpenCV Mat operations

### Architecture Improvements
- [ ] Create builder pattern for filter configuration
- [ ] Add filter parameter validation (window sizes, thresholds)
- [ ] Implement filter chain/pipeline abstraction
- [ ] Add debug/visualization traits for filters

### Missing Features
- [ ] Add filter parameter auto-tuning
- [ ] Implement adaptive filtering based on movement
- [ ] Add confidence scores to pose estimates
- [ ] Implement robust outlier rejection in pose estimation

### Code Maintainability
- [ ] Extract magic numbers to named constants (30 FPS assumption)
- [ ] Add configuration struct for default filter parameters
- [ ] Create factory methods with sensible defaults
- [ ] Add comprehensive examples in documentation

### Testing Improvements
- [ ] Add property-based tests for filters
- [ ] Test filter behavior with extreme values (infinity, NaN)
- [ ] Add benchmarks comparing filter performance
- [ ] Test thread safety of filters (Send + Sync traits)