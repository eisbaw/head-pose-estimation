//! Movement detection module for tracking head movement patterns.
//!
//! This module provides statistical analysis of head pose angles over time
//! to detect when the head is moving or stationary.

use std::collections::VecDeque;

/// Movement detector using statistical analysis
pub struct MovementDetector {
    window_size: usize,
    movement_threshold: f64,
    pitch_history: VecDeque<f64>,
    yaw_history: VecDeque<f64>,
}

impl MovementDetector {
    /// Create a new movement detector
    #[must_use]
    pub fn new(window_size: usize, movement_threshold: f64) -> Self {
        Self {
            window_size,
            movement_threshold,
            pitch_history: VecDeque::with_capacity(window_size),
            yaw_history: VecDeque::with_capacity(window_size),
        }
    }

    /// Update with new pose angles and detect if moving
    pub fn update(&mut self, pitch: f64, yaw: f64) -> bool {
        // Add to history
        if self.pitch_history.len() >= self.window_size {
            self.pitch_history.pop_front();
        }
        if self.yaw_history.len() >= self.window_size {
            self.yaw_history.pop_front();
        }

        self.pitch_history.push_back(pitch);
        self.yaw_history.push_back(yaw);

        // Need full window to detect movement
        if self.pitch_history.len() < self.window_size {
            return false;
        }

        // Calculate statistics
        let pitch_stats = Self::calculate_stats(&self.pitch_history);
        let yaw_stats = Self::calculate_stats(&self.yaw_history);

        // Check if moving based on standard deviation
        pitch_stats.std_dev > self.movement_threshold || yaw_stats.std_dev > self.movement_threshold
    }

    /// Get current statistics
    pub fn get_stats(&self) -> Option<(Statistics, Statistics)> {
        if self.pitch_history.len() < self.window_size {
            return None;
        }

        Some((
            Self::calculate_stats(&self.pitch_history),
            Self::calculate_stats(&self.yaw_history),
        ))
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.pitch_history.clear();
        self.yaw_history.clear();
    }

    /// Calculate statistics for a data window
    fn calculate_stats(data: &VecDeque<f64>) -> Statistics {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        let std_dev = variance.sqrt();

        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        Statistics {
            mean,
            std_dev,
            min,
            max,
            range,
        }
    }
}

/// Statistical summary of a data window
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    /// Mean value of the data
    pub mean: f64,
    /// Standard deviation of the data
    pub std_dev: f64,
    /// Minimum value in the window
    pub min: f64,
    /// Maximum value in the window
    pub max: f64,
    /// Range (max - min) of the data
    pub range: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_detection() {
        let mut detector = MovementDetector::new(5, 2.0);

        // Still head - small variations
        for i in 0..5 {
            let moving = detector.update(10.0 + (i as f64) * 0.1, 20.0 + (i as f64) * 0.1);
            // After filling window, should detect as not moving
            if i == 4 {
                assert!(!moving);
            }
        }

        // Moving head - large variations
        detector.reset();
        let angles = vec![(10.0, 20.0), (15.0, 25.0), (5.0, 15.0), (20.0, 30.0), (0.0, 10.0)];
        for (pitch, yaw) in angles {
            let _moving = detector.update(pitch, yaw);
            // Should eventually detect as moving
        }
    }

    #[test]
    fn test_statistics_calculation() {
        let data = VecDeque::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let stats = MovementDetector::calculate_stats(&data);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
        assert!((stats.std_dev - 1.4142135623730951).abs() < 1e-10);
    }
}
