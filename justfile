# Head Pose Estimation - Development Commands

# Default command - show available commands
default:
    @just --list

# Run all checks (build and test)
check: build test lint

# Set up development environment
setup:
    uv venv
    uv sync
    git lfs pull

# Run head pose estimation with webcam
run-cam cam="0":
    python main.py --cam {{cam}}

# Run head pose estimation with video file
run-video video:
    python main.py --video {{video}}

# Run with default webcam
run:
    python main.py

# Install dependencies
install:
    uv sync

# Update dependencies
update:
    uv lock --upgrade
    uv sync

# Clean up temporary files and caches
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name ".pytest_cache" -delete
    rm -rf .mypy_cache
    rm -rf .ruff_cache

# Download model files with Git LFS
download-models:
    git lfs pull

# Check if models are downloaded
check-models:
    @if [ -f "assets/face_detector.onnx" ] && [ -f "assets/face_landmarks.onnx" ]; then \
        echo "✓ Model files are present"; \
    else \
        echo "✗ Model files are missing. Run 'just download-models'"; \
        exit 1; \
    fi

# Enter Nix shell (if not already in it)
shell:
    nix-shell

# Run a quick test to ensure everything is working
test-setup: check-models
    python -c "import cv2, onnxruntime, numpy; print('✓ All imports successful')"

# Show Python and package versions
versions:
    python --version
    uv pip list | grep -E "(opencv|onnx|numpy)"

# === Rust Commands ===

# Build Rust project
build:
    cargo build

# Build Rust project in release mode
build-release:
    cargo build --release

# Run Rust tests
test:
    cargo test

# Run Rust linter and formatter
lint:
    cargo fmt -- --check
    cargo clippy -- -D warnings

# Format Rust code
fmt:
    cargo fmt

# Run Rust project with webcam
run-rust cam="0":
    cargo run -- --cam {{cam}}

# Run Rust project with video file
run-rust-video video:
    cargo run -- --video {{video}}

# Run Rust benchmarks
bench:
    cargo bench

# Clean Rust build artifacts
clean-rust:
    cargo clean

# Check Rust dependencies for vulnerabilities
audit:
    cargo audit

# Update Rust dependencies
update-rust:
    cargo update