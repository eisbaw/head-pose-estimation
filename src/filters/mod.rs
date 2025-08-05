//! Signal filtering algorithms for smoothing pose estimates.
//! 
//! This module provides various filtering algorithms to smooth noisy
//! head pose angle measurements, reducing jitter and improving stability.

/// Kalman filter implementation for optimal state estimation
pub mod kalman;

/// Moving average filter for simple smoothing
pub mod moving_average;

/// Median filter for outlier rejection
pub mod median;

/// Exponential filter for responsive smoothing
pub mod exponential;

/// Low-pass filters (first and second order) for frequency-based smoothing
pub mod low_pass;

/// Hampel filter for robust outlier detection and smoothing
pub mod hampel;

use crate::Result;

/// Trait for all cursor filters
pub trait CursorFilter: Send + Sync {
    /// Apply filter to input values
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64);
    
    /// Reset filter state
    fn reset(&mut self);
    
    /// Get filter name
    fn name(&self) -> &str;
}

/// No-op filter that passes through values unchanged
pub struct NoFilter;

impl CursorFilter for NoFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        (pitch, yaw)
    }
    
    fn reset(&mut self) {}
    
    fn name(&self) -> &str {
        "NoFilter"
    }
}

/// Create a cursor filter by type name
pub fn create_filter(filter_type: &str) -> Result<Box<dyn CursorFilter>> {
    match filter_type.to_lowercase().as_str() {
        "none" | "nofilter" => Ok(Box::new(NoFilter)),
        "kalman" => Ok(Box::new(kalman::KalmanFilter::new())),
        "moving_average" | "movingaverage" => Ok(Box::new(moving_average::MovingAverageFilter::new(5))),
        "median" => Ok(Box::new(median::MedianFilter::new(5))),
        "exponential" => Ok(Box::new(exponential::ExponentialFilter::new(0.5))),
        "lowpass" | "low_pass" => Ok(Box::new(low_pass::LowPassFilter::new(0.5))),
        "lowpass2" | "low_pass2" => Ok(Box::new(low_pass::SecondOrderLowPassFilter::new(30.0, 0.707))),
        "hampel" => Ok(Box::new(hampel::HampelFilter::new(5, 3.0))),
        _ => Err(crate::Error::FilterError(format!("Unknown filter type: {filter_type}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_filter() {
        let mut filter = NoFilter;
        let (pitch, yaw) = filter.apply(10.0, 20.0);
        assert_eq!(pitch, 10.0);
        assert_eq!(yaw, 20.0);
    }
    
    #[test]
    fn test_create_filter() {
        assert!(create_filter("none").is_ok());
        assert!(create_filter("kalman").is_ok());
        assert!(create_filter("unknown").is_err());
    }
}