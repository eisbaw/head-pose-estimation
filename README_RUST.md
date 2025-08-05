# Head Pose Estimation - Rust Implementation

Real-time human head pose estimation with ONNX Runtime and OpenCV, implemented in Rust for improved performance and safety.

![demo](doc/demo.gif)
![demo](doc/demo1.gif)

## Features

- **Real-time Performance**: Processes webcam feed at 30+ FPS
- **Advanced Filtering**: Multiple filter algorithms (Kalman, Moving Average, Exponential, etc.)
- **Movement Detection**: Intelligent head movement tracking
- **Cursor Control**: Use head movements to control mouse cursor (X11)
- **Memory Safe**: Written in Rust with zero unsafe code in core logic
- **Cross-platform Ready**: Designed for Linux with planned Windows/macOS support

## How It Works

The head pose estimation pipeline consists of three major steps:

1. **Face Detection**: SCRFD face detector provides bounding boxes for human faces
2. **Landmark Detection**: Deep learning model outputs 68 facial landmarks
3. **Pose Estimation**: PnP algorithm calculates head orientation from landmarks

## Prerequisites

### System Requirements

- Ubuntu 20.04+ (or other Linux distributions)
- OpenCV 4.5+ 
- ONNX Runtime 1.16+
- X11 (for cursor control features)
- Rust 1.70+

### Using Nix (Recommended)

If you have Nix installed, all dependencies are automatically managed:

```bash
nix-shell
```

### Manual Installation

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev \
    libclang-dev \
    libx11-dev \
    libxcb1-dev \
    pkg-config
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yinguobing/head-pose-estimation.git
cd head-pose-estimation
```

2. Download pre-trained models:
```bash
git lfs pull
```

Or download manually from the [release page](https://github.com/yinguobing/head-pose-estimation/releases).

3. Build the project:
```bash
cargo build --release
```

## Usage

### Basic Usage

Run with webcam (default):
```bash
cargo run --release
```

Run with specific webcam:
```bash
cargo run --release -- --cam 1
```

Run with video file:
```bash
cargo run --release -- --video path/to/video.mp4
```

### Command Line Options

```
head-pose-estimation [OPTIONS]

OPTIONS:
    -c, --cam <INDEX>              Webcam index [default: 0]
    -v, --video <PATH>             Path to video file
    -f, --filter <TYPE>            Filter type: moving_average, median, exponential, 
                                   kalman, low_pass, second_order_low_pass, hampel,
                                   no_filter [default: moving_average]
        --gui <MODE>               GUI display mode: all, pointers, cam, none 
                                   [default: all]
        --show-filters             Show all filter comparisons
        --use-normal-vector        Use normal vector projection for cursor
        --framerate <FPS>          Target framerate [default: 30]
        --conf-threshold <FLOAT>   Face detection confidence [default: 0.6]
        --iou-threshold <FLOAT>    Face detection IOU threshold [default: 0.5]
        --flip-x                   Mirror image horizontally
        --flip-y                   Mirror image vertically
        --invert-x                 Invert X axis for cursor control
        --invert-y                 Invert Y axis for cursor control
        --amplify-x <FLOAT>        X-axis amplification [default: 3.0]
        --amplify-y <FLOAT>        Y-axis amplification [default: 3.0]
    -h, --help                     Print help
    -V, --version                  Print version
```

### Cursor Control Modes

The application supports multiple cursor control modes:

- **Absolute Mode** (default): Direct mapping of head angles to cursor position
- **Relative Mode**: Press 'w' to set center point, cursor moves relative to center
- **Speed Mode**: Head angles control cursor velocity
- **Movement Mode**: Cursor only moves when head is moving

### Keyboard Shortcuts

- `q` - Quit application
- `w` - Toggle relative cursor mode / Set center point
- `f` - Cycle through filter types
- `g` - Toggle GUI display modes
- `m` - Toggle movement-based cursor control
- `n` - Toggle normal vector projection
- `s` - Toggle show all filters comparison

## Building from Source

### Development Build

```bash
cargo build
```

### Release Build (Optimized)

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Running with Just

If you have `just` installed:

```bash
just build    # Build in debug mode
just release  # Build in release mode
just test     # Run all tests
just run      # Run the application
```

## Project Structure

```
head-pose-estimation/
├── src/
│   ├── main.rs              # Application entry point
│   ├── lib.rs               # Library root with examples
│   ├── app.rs               # Main application logic
│   ├── face_detection.rs    # SCRFD face detector
│   ├── mark_detection.rs    # Facial landmark detector
│   ├── pose_estimation.rs   # PnP-based pose estimator
│   ├── filters/             # Signal filtering algorithms
│   ├── cursor_control.rs    # X11 cursor control
│   ├── movement_detector.rs # Movement detection logic
│   └── utils.rs             # Utility functions
├── assets/                  # Pre-trained ONNX models
├── tests/                   # Integration tests
└── Cargo.toml              # Project configuration
```

## Performance

The Rust implementation provides significant performance improvements over the Python version:

- **Startup Time**: < 1 second (vs 3-5 seconds for Python)
- **Memory Usage**: ~150MB (vs ~500MB for Python)  
- **Frame Rate**: Consistent 30+ FPS on modest hardware
- **Binary Size**: < 50MB standalone executable

## Filter Algorithms

The application includes several filtering algorithms for smoothing pose estimates:

1. **No Filter**: Raw pose estimates
2. **Moving Average**: Simple averaging over window
3. **Median Filter**: Robust to outliers
4. **Exponential Filter**: Adaptive smoothing
5. **Kalman Filter**: Optimal state estimation
6. **Low Pass Filter**: Frequency-based smoothing
7. **Second Order Low Pass**: Enhanced frequency filtering
8. **Hampel Filter**: Outlier-resistant median filter

## Troubleshooting

### ONNX Model Loading Issues

Ensure Git LFS is installed and models are downloaded:
```bash
git lfs pull
```

### OpenCV Errors

Verify OpenCV installation:
```bash
pkg-config --modversion opencv4
```

### X11 Cursor Control

For cursor control features, ensure you're running on X11 (not Wayland):
```bash
echo $XDG_SESSION_TYPE
```

## Development

### Running Clippy

```bash
cargo clippy --all-targets --all-features
```

### Formatting Code

```bash
cargo fmt
```

### Generating Documentation

```bash
cargo doc --open
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original Python implementation by [Yin Guobing](https://yinguobing.com)
- Face detector: [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) from InsightFace
- Training datasets: 300-W, 300-VW, LFPW, HELEN, AFW, IBUG
- 3D face model from [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] GPU acceleration for inference
- [ ] Windows and macOS support
- [ ] Wayland cursor control
- [ ] Multi-face tracking
- [ ] Configuration file support
- [ ] REST API for remote control