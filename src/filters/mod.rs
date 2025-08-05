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
/// 
/// # Supported filters:
/// - `none` or `nofilter` - No filtering
/// - `kalman` - Kalman filter
/// - `movingaverage:N` - Moving average with window size N (default: 5)
/// - `median:N` - Median filter with window size N (default: 5)
/// - `exponential:alpha` - Exponential filter with alpha (default: 0.5)
/// - `lowpass:cutoff` - First-order low-pass filter (default: 0.5)
/// - `secondorderlowpass:cutoff:damping` - Second-order low-pass (default: 30.0:0.707)
/// - `hampel:window:threshold` - Hampel filter (default: 5:3.0)
///
/// # Errors
///
/// Returns an error if:
/// - The filter type is not recognized
/// - The filter parameters are invalid
pub fn create_filter(filter_type: &str) -> Result<Box<dyn CursorFilter>> {
    let parts: Vec<&str> = filter_type.split(':').collect();
    let filter_name = parts[0].to_lowercase();
    
    match filter_name.as_str() {
        "none" | "nofilter" => Ok(Box::new(NoFilter)),
        "kalman" => Ok(Box::new(kalman::KalmanFilter::new())),
        "movingaverage" | "moving_average" => {
            let window_size = if parts.len() > 1 {
                let val = parts[1].parse::<usize>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid window size: {}", parts[1])))?;
                if val == 0 {
                    return Err(crate::Error::FilterError("Window size must be greater than 0".to_string()));
                }
                val
            } else {
                5
            };
            Ok(Box::new(moving_average::MovingAverageFilter::new(window_size)))
        }
        "median" => {
            let window_size = if parts.len() > 1 {
                let val = parts[1].parse::<usize>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid window size: {}", parts[1])))?;
                if val == 0 {
                    return Err(crate::Error::FilterError("Window size must be greater than 0".to_string()));
                }
                if val % 2 == 0 {
                    return Err(crate::Error::FilterError(format!("Median filter window size must be odd, got {}", val)));
                }
                val
            } else {
                5
            };
            Ok(Box::new(median::MedianFilter::new(window_size)))
        }
        "exponential" => {
            let alpha = if parts.len() > 1 {
                let val = parts[1].parse::<f64>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid alpha value: {}", parts[1])))?;
                if val <= 0.0 || val > 1.0 {
                    return Err(crate::Error::FilterError(format!("Alpha must be in (0, 1], got {val}")));
                }
                val
            } else {
                0.5
            };
            Ok(Box::new(exponential::ExponentialFilter::new(alpha)))
        }
        "lowpass" | "low_pass" => {
            let cutoff = if parts.len() > 1 {
                let val = parts[1].parse::<f64>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid cutoff frequency: {}", parts[1])))?;
                if val <= 0.0 || val > 1.0 {
                    return Err(crate::Error::FilterError(format!("Cutoff must be in (0, 1], got {val}")));
                }
                val
            } else {
                0.5
            };
            Ok(Box::new(low_pass::LowPassFilter::new(cutoff)))
        }
        "secondorderlowpass" | "lowpass2" | "low_pass2" => {
            let cutoff = if parts.len() > 1 {
                parts[1].parse::<f64>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid cutoff frequency: {}", parts[1])))?
            } else {
                30.0
            };
            let damping = if parts.len() > 2 {
                parts[2].parse::<f64>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid damping ratio: {}", parts[2])))?
            } else {
                0.707
            };
            Ok(Box::new(low_pass::SecondOrderLowPassFilter::new(cutoff, damping)))
        }
        "hampel" => {
            let window_size = if parts.len() > 1 {
                let val = parts[1].parse::<usize>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid window size: {}", parts[1])))?;
                if val == 0 {
                    return Err(crate::Error::FilterError("Window size must be greater than 0".to_string()));
                }
                val
            } else {
                5
            };
            let threshold = if parts.len() > 2 {
                let val = parts[2].parse::<f64>()
                    .map_err(|_| crate::Error::FilterError(format!("Invalid threshold: {}", parts[2])))?;
                if val < 0.0 {
                    return Err(crate::Error::FilterError(format!("Threshold must be non-negative, got {val}")));
                }
                val
            } else {
                3.0
            };
            Ok(Box::new(hampel::HampelFilter::new(window_size, threshold)))
        }
        _ => Err(crate::Error::FilterError(format!("Unknown filter type: {filter_name}"))),
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

    #[test]
    fn test_create_filter_with_params() {
        // Test parameterized filters
        assert!(create_filter("movingaverage:10").is_ok());
        assert!(create_filter("median:7").is_ok());
        assert!(create_filter("exponential:0.8").is_ok());
        assert!(create_filter("lowpass:0.3").is_ok());
        assert!(create_filter("secondorderlowpass:20:0.5").is_ok());
        assert!(create_filter("hampel:9:2.5").is_ok());
        
        // Test invalid parameters
        assert!(create_filter("movingaverage:abc").is_err());
        assert!(create_filter("exponential:2.0").is_err()); // Alpha should be in (0, 1]
        assert!(create_filter("median:0").is_err()); // Window size must be > 0
    }
}
