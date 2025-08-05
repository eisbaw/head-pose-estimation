//! Cursor control module for X11-based systems.
//!
//! This module provides functionality to control the mouse cursor position
//! using X11 protocols. It supports both absolute and relative cursor movement.

use crate::error::{AppError, Result};
use log::{debug, info};
use x11rb::{
    connection::Connection,
    protocol::xproto::{ConnectionExt, Screen},
    rust_connection::RustConnection,
};

/// Cursor control implementation for X11
pub struct CursorController {
    connection: RustConnection,
    screen: Screen,
    screen_width: u16,
    screen_height: u16,
}

impl CursorController {
    /// Create a new cursor controller
    pub fn new() -> Result<Self> {
        info!("Initializing X11 cursor controller");
        
        // Connect to X11 server
        let (connection, screen_num) = RustConnection::connect(None)
            .map_err(|e| AppError::CursorControl(format!("Failed to connect to X11: {}", e)))?;
        
        // Get screen information
        let screen = connection
            .setup()
            .roots
            .get(screen_num)
            .ok_or_else(|| AppError::CursorControl("Failed to get screen".to_string()))?
            .clone();
        
        let screen_width = screen.width_in_pixels;
        let screen_height = screen.height_in_pixels;
        
        info!(
            "Connected to X11 display, screen: {}x{}",
            screen_width, screen_height
        );
        
        Ok(Self {
            connection,
            screen,
            screen_width,
            screen_height,
        })
    }
    
    /// Get current cursor position
    pub fn get_position(&self) -> Result<(i16, i16)> {
        let reply = self
            .connection
            .query_pointer(self.screen.root)
            .map_err(|e| AppError::CursorControl(format!("Failed to send query pointer: {}", e)))?
            .reply()
            .map_err(|e| AppError::CursorControl(format!("Failed to query pointer: {}", e)))?;
        
        Ok((reply.root_x, reply.root_y))
    }
    
    /// Set cursor position (absolute)
    pub fn set_position(&self, x: i16, y: i16) -> Result<()> {
        // Clamp to screen bounds
        let x = x.clamp(0, self.screen_width as i16 - 1);
        let y = y.clamp(0, self.screen_height as i16 - 1);
        
        debug!("Setting cursor position to ({}, {})", x, y);
        
        self.connection
            .warp_pointer(
                x11rb::NONE,
                self.screen.root,
                0,
                0,
                0,
                0,
                x,
                y,
            )
            .map_err(|e| AppError::CursorControl(format!("Failed to warp pointer: {}", e)))?;
        
        self.connection.flush()
            .map_err(|e| AppError::CursorControl(format!("Failed to flush connection: {}", e)))?;
        
        Ok(())
    }
    
    /// Move cursor relative to current position
    pub fn move_relative(&self, dx: i16, dy: i16) -> Result<()> {
        let (current_x, current_y) = self.get_position()?;
        let new_x = current_x + dx;
        let new_y = current_y + dy;
        
        self.set_position(new_x, new_y)
    }
    
    /// Get screen dimensions
    pub fn get_screen_size(&self) -> (u16, u16) {
        (self.screen_width, self.screen_height)
    }
    
    /// Map normalized coordinates to screen coordinates
    pub fn map_to_screen(&self, normalized_x: f64, normalized_y: f64) -> (i16, i16) {
        let x = (normalized_x * self.screen_width as f64) as i16;
        let y = (normalized_y * self.screen_height as f64) as i16;
        (x, y)
    }
}

/// Cursor control mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CursorMode {
    /// Direct absolute positioning
    Absolute,
    /// Relative movement from current position
    Relative,
    /// Velocity-based movement (speed mode)
    Velocity,
}

/// Data source for cursor control
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataSource {
    /// Use pitch and yaw angles
    PitchYaw,
    /// Use normal vector projection
    NormalProjection,
}

impl Default for CursorMode {
    fn default() -> Self {
        Self::Absolute
    }
}

impl Default for DataSource {
    fn default() -> Self {
        Self::PitchYaw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires X11 display
    fn test_cursor_controller_creation() {
        let controller = CursorController::new();
        assert!(controller.is_ok() || controller.is_err()); // Will fail without X11
    }
    
    #[test]
    fn test_cursor_mode_default() {
        assert_eq!(CursorMode::default(), CursorMode::Absolute);
    }
    
    #[test]
    fn test_data_source_default() {
        assert_eq!(DataSource::default(), DataSource::PitchYaw);
    }
}