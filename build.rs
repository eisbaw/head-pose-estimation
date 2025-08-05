//! Build script for detecting system dependencies and providing installation guidance.
//!
//! This script checks for required system libraries (OpenCV, X11, pkg-config) and
//! provides helpful error messages if they are missing.

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check for OpenCV
    check_opencv();

    // Check for X11 (required for cursor control on Linux)
    check_x11();

    // Check for pkg-config
    check_pkg_config();

    // Print detected environment
    println!(
        "cargo:rustc-env=BUILD_TARGET={}",
        env::var("TARGET").unwrap_or_default()
    );
    println!("cargo:rustc-env=BUILD_HOST={}", env::var("HOST").unwrap_or_default());
}

fn check_opencv() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=OPENCV_LINK_PATHS");
    println!("cargo:rerun-if-env-changed=OPENCV_INCLUDE_PATHS");

    // Try to find OpenCV using pkg-config
    let output = Command::new("pkg-config").args(["--modversion", "opencv4"]).output();

    match output {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=Found OpenCV version: {}", version.trim());
        }
        _ => {
            // Try opencv instead of opencv4
            let output = Command::new("pkg-config").args(["--modversion", "opencv"]).output();

            match output {
                Ok(output) if output.status.success() => {
                    let version = String::from_utf8_lossy(&output.stdout);
                    println!("cargo:warning=Found OpenCV version: {}", version.trim());
                }
                _ => {
                    println!("cargo:warning=OpenCV not found via pkg-config. Make sure OpenCV is installed.");
                    println!("cargo:warning=On Ubuntu: sudo apt-get install libopencv-dev");
                    println!("cargo:warning=On macOS: brew install opencv");
                    println!("cargo:warning=On NixOS: Use the provided shell.nix");
                }
            }
        }
    }
}

fn check_x11() {
    // Only check on Linux
    if env::var("TARGET").unwrap_or_default().contains("linux") {
        let output = Command::new("pkg-config").args(["--exists", "x11"]).output();

        match output {
            Ok(output) if output.status.success() => {
                println!("cargo:warning=Found X11 libraries");
            }
            _ => {
                println!("cargo:warning=X11 libraries not found. Cursor control features will not work.");
                println!("cargo:warning=On Ubuntu: sudo apt-get install libx11-dev");
                println!("cargo:warning=On NixOS: Use the provided shell.nix");
            }
        }
    }
}

fn check_pkg_config() {
    let output = Command::new("pkg-config").arg("--version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=Found pkg-config version: {}", version.trim());
        }
        _ => {
            println!("cargo:warning=pkg-config not found. This is required to find system libraries.");
            println!("cargo:warning=On Ubuntu: sudo apt-get install pkg-config");
            println!("cargo:warning=On macOS: brew install pkg-config");
            println!("cargo:warning=On NixOS: Use the provided shell.nix");
        }
    }
}
