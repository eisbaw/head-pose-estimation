# Product Requirements Document: Head Pose Estimation Rust Port

## Executive Summary

This document outlines the requirements and implementation plan for converting the existing Python-based head pose estimation application to Rust. The application performs real-time human head pose estimation using face detection, facial landmark detection, and pose calculation algorithms. The Rust implementation will maintain feature parity while potentially offering improved performance, memory safety, and deployment flexibility.

## Project Overview

### Current State
- **Language**: Python 3.9+
- **Key Libraries**: OpenCV, ONNX Runtime, NumPy
- **Architecture**: Modular design with separate components for face detection, landmark detection, and pose estimation
- **Features**: Real-time webcam/video processing, multiple cursor control modes, various smoothing filters
- **Platform**: Linux with X11 (xdotool dependency for cursor control)

### Target State
- **Language**: Rust
- **Performance**: Equal or better than Python implementation
- **Safety**: Memory-safe implementation with no undefined behavior
- **Deployment**: Single binary with minimal runtime dependencies
- **Models**: Continue using existing ONNX models (face_detector.onnx, face_landmarks.onnx) without modification

## Core Components Analysis

### 1. Face Detection Module (`face_detection.py`)
- **Purpose**: Detect human faces using SCRFD model
- **Key Features**:
  - ONNX model inference
  - Non-maximum suppression (NMS)
  - Bounding box refinement
  - Support for multiple faces (though only first is used)
- **Rust Dependencies**: `ort` (ONNX Runtime), custom NMS implementation

### 2. Facial Landmark Detection (`mark_detection.py`)
- **Purpose**: Detect 68 facial landmarks
- **Key Features**:
  - ONNX model inference
  - Image preprocessing (resize, color conversion)
  - Batch processing support
- **Rust Dependencies**: `ort`, `image` or `opencv` for preprocessing

### 3. Pose Estimation (`pose_estimation.py`)
- **Purpose**: Calculate head pose from facial landmarks
- **Key Features**:
  - PnP (Perspective-n-Point) problem solving
  - Euler angle extraction
  - 3D visualization capabilities
  - Camera matrix calculations
- **Rust Dependencies**: `opencv` or `cv` crate for PnP solver, `nalgebra` for matrix operations

### 4. Cursor Filters (`cursor_filters.py`)
- **Purpose**: Smooth cursor movement with various filter algorithms
- **Implementations**:
  - Kalman filter
  - Moving average
  - Median filter
  - Exponential smoothing
  - Low-pass filters (1st and 2nd order)
  - Hampel filter
- **Rust Dependencies**: `nalgebra` for matrix operations, custom DSP implementations

### 5. Movement Detection (`movement_detector.py`)
- **Purpose**: Detect head movement vs. stillness
- **Key Features**:
  - Statistical analysis (std deviation, range)
  - Sliding window buffer
- **Rust Dependencies**: Standard library collections, basic statistics

### 6. Main Application (`main.py`)
- **Purpose**: Orchestrate all components and handle UI/UX
- **Key Features**:
  - Video capture and display
  - Command-line argument parsing
  - Multiple GUI modes
  - Cursor control (absolute/relative)
  - Keyboard input detection
  - Multi-threading for smooth cursor updates

## Rust Implementation Strategy

### Phase 1: Core Libraries Setup
1. **Create Cargo project structure**
   ```toml
   [dependencies]
   opencv = "0.88"  # For video capture and computer vision
   ort = "1.16"     # ONNX Runtime bindings
   nalgebra = "0.32" # Linear algebra
   clap = "4.4"     # CLI argument parsing
   anyhow = "1.0"   # Error handling
   x11rb = "0.12"   # X11 interaction (cursor control)
   ```

2. **Module structure**:
   ```
   src/
   ├── main.rs
   ├── face_detection.rs
   ├── mark_detection.rs
   ├── pose_estimation.rs
   ├── filters/
   │   ├── mod.rs
   │   ├── kalman.rs
   │   ├── moving_average.rs
   │   └── ...
   ├── movement_detector.rs
   └── utils.rs
   ```

### Phase 2: Component Implementation Order

1. **Utils and Basic Infrastructure** (Week 1)
   - Image preprocessing functions
   - Bounding box operations
   - Error handling framework

2. **ONNX Model Integration** (Week 2)
   - Face detection module
   - Landmark detection module
   - Model loading and inference
   - Use existing ONNX files from `assets/` directory without modification

3. **Pose Estimation** (Week 3)
   - PnP solver integration
   - Euler angle calculations
   - Camera matrix setup

4. **Filters Implementation** (Week 4)
   - Port all filter algorithms
   - Create filter trait/interface
   - Factory pattern for filter creation

5. **Movement Detection** (Week 5)
   - Statistical calculations
   - Circular buffer implementation

6. **Main Application Logic** (Week 6-7)
   - Video capture loop
   - GUI windows (using OpenCV highgui)
   - X11 cursor control
   - Threading for cursor updates
   - Keyboard input handling

7. **Testing and Optimization** (Week 8)
   - Performance benchmarking
   - Memory profiling
   - Cross-platform considerations

### Phase 3: Rust-Specific Enhancements

1. **Performance Optimizations**:
   - SIMD operations for filter calculations
   - Parallel processing where applicable
   - Zero-copy frame processing

2. **Safety Improvements**:
   - Type-safe state machines for application modes
   - Compile-time guarantees for filter parameters
   - Safe FFI wrappers for X11 operations

3. **Deployment**:
   - Static linking where possible
   - Cross-compilation support
   - Minimal runtime dependencies

## Technical Challenges and Solutions

### 1. ONNX Runtime Integration
- **Challenge**: Efficient tensor manipulation
- **Solution**: Use `ndarray` with ONNX Runtime for zero-copy operations where possible
- **Model Compatibility**: The Rust implementation will load and use the exact same ONNX model files (`assets/face_detector.onnx` and `assets/face_landmarks.onnx`) without any modifications or conversions

### 2. OpenCV Dependency
- **Challenge**: OpenCV can be heavy and complex to build
- **Solution**: Consider using `opencv` crate with pre-built bindings, or implement critical functions natively

### 3. X11 Cursor Control
- **Challenge**: Safe X11 interaction
- **Solution**: Use `x11rb` for safe X11 bindings, implement xdotool functionality natively

### 4. Real-time Performance
- **Challenge**: Maintaining 30+ FPS with all filters
- **Solution**: Profile critical paths, use const generics for compile-time optimizations

### 5. GUI Compatibility
- **Challenge**: Matching OpenCV's highgui functionality
- **Solution**: Use OpenCV's Rust bindings initially, consider native alternatives later

## Migration Path

### Step 1: Proof of Concept
- Implement basic face detection and display
- Verify ONNX model loading and inference
- Establish video capture pipeline

### Step 2: Feature Parity
- Port all modules maintaining exact behavior
- Ensure all command-line arguments work identically
- Match filter implementations precisely

### Step 3: Testing Strategy
- Unit tests for each filter algorithm
- Integration tests for pose estimation accuracy
- Performance benchmarks against Python version
- End-to-end testing with various video inputs

### Step 4: Optimization
- Profile and optimize hot paths
- Reduce memory allocations
- Implement SIMD where beneficial

## Success Criteria

1. **Functional Requirements**:
   - All command-line options work identically
   - Filter outputs match Python implementation within floating-point tolerance
   - Cursor control functions correctly on Linux/X11

2. **Performance Requirements**:
   - Equal or better FPS than Python version
   - Lower memory usage
   - Faster startup time

3. **Quality Requirements**:
   - No unsafe code outside of FFI boundaries
   - Comprehensive error handling
   - Clear documentation
   - Reproducible builds

## Dependencies and External Requirements

### Rust Crate Dependencies
- `opencv` (0.88+): Video capture and display
- `ort` (1.16+): ONNX Runtime bindings
- `nalgebra` (0.32+): Linear algebra operations
- `clap` (4.4+): Command-line parsing
- `anyhow` (1.0+): Error handling
- `x11rb` (0.12+): X11 interaction
- `rayon` (1.7+): Parallel processing
- `log` + `env_logger`: Logging

### System Requirements
- Rust 1.70+ (for stable const generics)
- ONNX Runtime 1.16+ system library
- OpenCV 4.5+ system library
- X11 development libraries (Linux)
- Existing ONNX model files:
  - `assets/face_detector.onnx` (SCRFD face detection model)
  - `assets/face_landmarks.onnx` (68-point facial landmark model)
  - `assets/model.txt` (3D face model points)

### Build Requirements
- `pkg-config` for library discovery
- `cmake` for OpenCV if building from source
- `clang` for bindgen (opencv-rust)

## Project Timeline

- **Week 1-2**: Setup and basic infrastructure
- **Week 3-4**: Core detection modules
- **Week 5-6**: Filters and pose estimation
- **Week 7-8**: Main application and integration
- **Week 9-10**: Testing and optimization
- **Week 11-12**: Documentation and deployment

Total estimated time: 12 weeks for full feature parity with optimizations

## Risk Assessment

1. **OpenCV Rust bindings stability**: Medium risk - fallback to writing critical bindings
2. **ONNX Runtime compatibility**: Low risk - crate is well-maintained
3. **Performance regression**: Low risk - Rust typically outperforms Python
4. **X11 platform lock-in**: Medium risk - design with abstraction for future Wayland support

## Conclusion

The Rust port of the head pose estimation application is feasible and will likely result in improved performance, safety, and deployment characteristics. The modular architecture of the Python version translates well to Rust's module system. The main challenges involve managing external dependencies and ensuring numerical accuracy in filter implementations. With careful planning and phased implementation, this project can deliver a superior version of the application while maintaining full backward compatibility.