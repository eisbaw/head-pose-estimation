# Head Pose Estimation Rust Port - Task List

## Phase 1: Project Setup and Infrastructure

### Week 1: Basic Setup
- [x] DONE: Create new Rust project with Cargo
- [ ] Set up workspace structure with multiple crates if needed
- [x] DONE: Add initial dependencies to Cargo.toml
- [x] DONE: Create module structure matching Python architecture
- [x] DONE: Set up error handling with anyhow/thiserror
- [x] DONE: Configure logging with env_logger
- [x] DONE: Create build script for system dependency detection
- [x] DONE: Set up CI/CD pipeline (GitHub Actions)
- [x] DONE: Add rustfmt.toml and clippy configuration (in Cargo.toml)

### Week 2: Core Infrastructure
- [x] DONE: Implement basic image type conversions (OpenCV Mat <-> ndarray)
- [x] DONE: Create bounding box struct and operations (using opencv::core::Rect)
- [x] DONE: Port `refine()` function from utils.py
- [x] DONE: Implement basic geometry helpers
- [x] DONE: Create trait for cursor filters
- [x] DONE: Set up ONNX Runtime initialization
- [x] DONE: Add configuration struct for application settings
- [x] DONE: Implement command-line argument parsing with clap

## Phase 2: ONNX Model Integration

### Week 3: Face Detection Module
- [x] DONE: Create FaceDetector struct
- [x] DONE: Implement ONNX model loading for face_detector.onnx
- [x] DONE: Port image preprocessing for face detection
- [x] DONE: Implement forward pass inference
- [x] DONE: Port distance2bbox function
- [x] DONE: Port distance2kps function
- [x] DONE: Implement NMS (Non-Maximum Suppression)
- [x] DONE: Add face detection visualization
- [x] DONE: Write unit tests for face detection (test_distance_to_bbox)

### Week 4: Landmark Detection Module
- [x] DONE: Create MarkDetector struct
- [x] DONE: Implement ONNX model loading for face_landmarks.onnx
- [x] DONE: Port image preprocessing (resize, color conversion)
- [x] DONE: Implement batch inference support
- [x] DONE: Convert landmark output format
- [x] DONE: Add landmark visualization
- [x] DONE: Write unit tests for landmark detection

## Phase 3: Pose Estimation

### Week 5: Pose Estimation Core
- [x] DONE: Create PoseEstimator struct
- [x] DONE: Load 3D model points from assets/model.txt
- [x] DONE: Implement camera matrix initialization
- [x] DONE: Port PnP solver using OpenCV
- [x] DONE: Implement Euler angle extraction
- [ ] Port pose visualization (3D box drawing)
- [x] DONE: Implement draw_axes functionality
- [x] DONE: Implement draw_normal_vector functionality
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
- [x] DONE: Add benchmarks for filters
- [x] DONE: Ensure numerical accuracy matches Python

## Phase 5: Application Features

### Week 8: Movement Detection and Utilities
- [x] DONE: Create MovementDetector struct
- [x] DONE: Implement sliding window buffer
- [x] DONE: Port statistical calculations
- [x] DONE: Add movement detection logic
- [x] DONE: Implement debug statistics
- [x] DONE: Create utility functions for screen resolution
- [x] DONE: Add X11 cursor control functions
- [x] DONE: Implement keyboard state detection

### Week 9: Video Processing Pipeline
- [x] DONE: Set up OpenCV VideoCapture
- [x] DONE: Implement frame reading loop
- [x] DONE: Add webcam buffer optimization
- [x] DONE: Implement image flipping/inversion options
- [x] DONE: Add brightness adjustment
- [x] DONE: Create frame timing/FPS counter
- [x] DONE: Implement graceful shutdown
- [x] DONE: Add video file support

## Phase 6: Main Application

### Week 10: Core Application Logic
- [x] DONE: Implement main application struct
- [x] DONE: Port argument parsing logic
- [x] DONE: Create GUI window management
- [x] DONE: Implement cursor position mapping (angles to pixels)
- [x] DONE: Implement normal vector projection mode
- [x] DONE: Add multiple filter visualization mode
- [x] DONE: Implement GUI mode selection (all/pointers/cam/none)
- [x] DONE: Add legend and UI text rendering

### Week 11: Cursor Control Features
- [x] DONE: Implement absolute cursor control mode
- [x] DONE: Implement relative cursor control mode
- [x] DONE: Add 'w' key detection for relative mode
- [x] DONE: Implement location vector mode
- [x] DONE: Implement speed vector mode
- [x] DONE: Create cursor update thread for speed mode
- [x] DONE: Add movement-based cursor control
- [x] DONE: Test cursor control on X11

## Phase 7: Testing and Optimization

### Week 12: Integration and Testing
- [x] DONE: Create integration tests for full pipeline
- [x] DONE: Add performance benchmarks vs Python
- [ ] Profile memory usage
- [ ] Optimize hot paths
- [ ] Add SIMD optimizations where beneficial
- [ ] Test with various video inputs
- [x] DONE: Ensure filter outputs match Python within tolerance
- [ ] Create test suite for cursor control

### Week 13: Polish and Documentation
- [x] DONE: Write comprehensive README
- [x] DONE: Add inline documentation
- [x] DONE: Create usage examples
- [x] DONE: Document build instructions
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
- [x] DONE: Add configuration file support
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
- [x] DONE: Test video file processing

### Performance Tests
- [ ] Benchmark vs Python implementation
- [ ] Measure memory usage
- [ ] Test real-time performance (30+ FPS)
- [ ] Profile CPU usage
- [ ] Test with different video resolutions

### Compatibility Tests
- [x] DONE: Test with original ONNX models
- [x] DONE: Test all command-line arguments
- [x] DONE: Test filter output accuracy
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

- [x] DONE: Set up shell.nix with Rust toolchain
- [x] DONE: Add ONNX Runtime to shell.nix
- [x] DONE: Configure OpenCV in shell.nix
- [x] DONE: Create justfile with build commands
- [ ] Add development dependencies
- [ ] Set up pre-commit hooks
- [x] DONE: Configure rustfmt and clippy

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
- [x] DONE: Fix inline format string warnings (use {variable} syntax)
- [x] DONE: Remove unnecessary Result wrappers
- [x] DONE: Fix unused self arguments - refactor to associated functions
- [x] DONE: Fix needless_pass_by_value warnings
- [x] DONE: Fix type_complexity warning with type aliases
- [x] DONE: Replace manual range loops with iterators
- [x] DONE: Fix cast_lossless warnings - use From trait
- [x] DONE: Fix missing documentation for constants
- [x] DONE: Add comprehensive error handling tests
- [x] DONE: Add property-based tests with proptest
- [x] DONE: Fix filter parameter validation to prevent panics

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
- [x] DONE: Add comprehensive error handling tests
- [x] DONE: Add property-based tests for numeric conversions
- [x] DONE: Test edge cases for safe_cast functions

## Code Quality Issues (Found by Clippy)

### High Priority
- [x] DONE: Fix potential panic in KalmanFilter matrix inversion (already uses try_inverse)
- [x] DONE: Fix suspicious operation grouping in SecondOrderLowPassFilter (line 96)
- [x] DONE: Add proper error handling for all unwrap() calls in tests
- [x] DONE: Handle usize to i32 casts that may wrap on 32-bit systems
- [x] DONE: Add more backticks to documentation items (OpenCV, PnP, window_size)
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
- [x] DONE: Test MovementDetector with edge cases
- [x] DONE: Test PoseEstimator::parse_model_points with invalid input
- [x] DONE: Test filter creation with parameters
- [x] DONE: Test filter extreme values (infinity, NaN)
- [x] DONE: Test filter convergence and impulse response
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
- [x] DONE: Extract magic numbers to named constants (30 FPS assumption)
- [ ] Add configuration struct for default filter parameters
- [ ] Create factory methods with sensible defaults
- [ ] Add comprehensive examples in documentation

### Testing Improvements
- [ ] Add property-based tests for filters
- [ ] Test filter behavior with extreme values (infinity, NaN)
- [x] DONE: Add benchmarks comparing filter performance
- [ ] Test thread safety of filters (Send + Sync traits)

## Code Smells Found by Clippy (New)

### Critical Issues
- [x] DONE: Fix many usize to i32 casts that may wrap on 32-bit systems
- [x] DONE: Fix u16 to i16 casts that may wrap
- [x] DONE: Fix complex type in face_detection forward() return type
- [x] DONE: Replace manual range loops with iterators

### Medium Priority
- [x] DONE: Use inline format variables instead of positional parameters
- [x] DONE: Derive Eq for enums that derive PartialEq  
- [x] DONE: Make const-eligible functions const
- [x] DONE: Use f64::from() for lossless casts (added allow attributes where precision loss is acceptable)
- [x] DONE: Remove unused self parameters or refactor to associated functions
- [x] DONE: Fix similar variable names (stats_text vs status_text)

### Performance Improvements
- [ ] Replace Vec<u8> allocation with iterators in face detection
- [ ] Use slice patterns instead of indexing
- [ ] Consider SmallVec for small collections
- [x] DONE: Avoid unnecessary clones with .copied()

### Code Quality
- [x] DONE: Add error context instead of generic "Failed to X" messages
- [x] DONE: Use more specific error types (used allow attributes for acceptable cases)
- [x] DONE: Document panics in public APIs (added # Errors sections)
- [x] DONE: Add examples to public functions
- [x] DONE: Use type aliases for complex types

## New Code Quality Issues Found (2025-08-05)

### Error Handling Improvements
- [x] DONE: Create domain-specific error variants instead of generic ModelError
- [ ] Add error recovery strategies for non-fatal errors
- [ ] Implement retry logic for transient failures
- [ ] Add structured logging with error codes

### Code Organization
- [ ] Split large modules (app.rs, face_detection.rs) into smaller files
- [x] DONE: Extract constants to a dedicated configuration module
- [ ] Create trait for video sources (webcam, file, stream)
- [ ] Implement builder pattern for complex structs

### Documentation
- [ ] Add comprehensive examples in lib.rs
- [ ] Document performance characteristics of filters
- [ ] Add architecture diagram in README
- [ ] Create migration guide from Python version

### Testing Gaps
- [x] DONE: Add property-based tests for numeric conversions
- [x] DONE: Test edge cases for safe_cast functions
- [ ] Add fuzz testing for ONNX model inputs
- [ ] Create benchmarks for critical paths

### Performance Optimization Opportunities
- [ ] Investigate SIMD for filter calculations
- [ ] Profile memory allocations in hot paths
- [ ] Consider pre-allocating buffers for video frames
- [ ] Optimize anchor center generation with const evaluation