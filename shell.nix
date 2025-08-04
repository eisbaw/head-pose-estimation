{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    uv
    just
    git
    git-lfs
    
    # Rust toolchain
    rustc
    cargo
    rustfmt
    clippy
    rust-analyzer
    pkg-config
    
    # Build tools
    cmake
    clang
    llvmPackages.libclang
    llvmPackages.llvm
    
    # System dependencies needed by OpenCV and numpy
    stdenv.cc.cc.lib
    zlib
    libGL
    libGLU
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libICE
    xorg.libSM
    xorg.libXxf86vm
    xorg.libXi
    glib
    gtk3
    ffmpeg
    
    # Additional libraries that might be needed
    freetype
    fontconfig
    libpng
    libjpeg
    libtiff
    libwebp
    openblas
    lapack
    
    # ONNX Runtime - required for Rust port
    onnxruntime
    
    # OpenCV for Rust
    opencv
    
    # X11 development libraries for cursor control
    xorg.libXtst
    xdotool
  ];

  shellHook = ''
    # Set library path for all dependencies
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
      stdenv.cc.cc.lib
      zlib
      libGL
      libGLU
      xorg.libX11
      xorg.libXext
      xorg.libXrender
      xorg.libICE
      xorg.libSM
      xorg.libXxf86vm
      xorg.libXi
      xorg.libXtst
      glib
      gtk3
      ffmpeg
      freetype
      fontconfig
      libpng
      libjpeg
      libtiff
      libwebp
      openblas
      lapack
      onnxruntime
      opencv
    ])}:$LD_LIBRARY_PATH"
    
    # Set PKG_CONFIG_PATH for Rust dependencies
    export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" [
      pkgs.opencv
      pkgs.onnxruntime
      pkgs.xorg.libX11
      pkgs.xorg.libXtst
    ]}:$PKG_CONFIG_PATH"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      uv venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install Python dependencies
    uv sync
    
    # Set up Rust environment
    export OPENCV_PKG_CONFIG_PATH="${pkgs.opencv}/lib/pkgconfig"
    export ORT_LIB_DIR="${pkgs.onnxruntime}/lib"
    export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
    export LLVM_CONFIG_PATH="${pkgs.llvmPackages.llvm}/bin/llvm-config"
    
    echo "Development environment ready for both Python and Rust"
    echo "Rust toolchain: $(rustc --version)"
    echo "OpenCV: ${pkgs.opencv.version}"
    echo "ONNX Runtime: ${pkgs.onnxruntime.version}"
  '';
}