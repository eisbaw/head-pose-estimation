{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    uv
    just
    git
    git-lfs
    
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
    ])}:$LD_LIBRARY_PATH"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      uv venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    uv sync
  '';
}