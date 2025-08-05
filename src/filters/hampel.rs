use super::CursorFilter;
use std::collections::VecDeque;

/// Hampel filter for outlier removal
pub struct HampelFilter {
    window_size: usize,
    threshold: f64,
    pitch_buffer: VecDeque<f64>,
    yaw_buffer: VecDeque<f64>,
}

impl HampelFilter {
    /// Create a new Hampel filter
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is not odd or is zero, or if threshold is negative
    #[must_use]
    pub fn new(window_size: usize, threshold: f64) -> Self {
        assert!(window_size > 0, "Window size must be greater than 0");
        assert!(window_size % 2 == 1, "Window size must be odd, got {}", window_size);
        assert!(threshold >= 0.0, "Threshold must be non-negative, got {}", threshold);
        Self {
            window_size,
            threshold,
            pitch_buffer: VecDeque::with_capacity(window_size),
            yaw_buffer: VecDeque::with_capacity(window_size),
        }
    }

    fn hampel_filter(values: &VecDeque<f64>, new_value: f64, threshold: f64) -> f64 {
        if values.is_empty() {
            return new_value;
        }

        // Calculate median
        let mut sorted: Vec<f64> = values.iter().copied().collect();
        sorted.push(new_value);
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate median absolute deviation (MAD)
        let mut deviations: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if deviations.len() % 2 == 0 {
            (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
        } else {
            deviations[deviations.len() / 2]
        };

        // Robust standard deviation estimate
        let sigma = 1.4826 * mad;

        // Check if new value is an outlier
        if (new_value - median).abs() > threshold * sigma {
            median // Replace outlier with median
        } else {
            new_value
        }
    }
}

impl CursorFilter for HampelFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        // Apply Hampel filter
        let filtered_pitch = Self::hampel_filter(&self.pitch_buffer, pitch, self.threshold);
        let filtered_yaw = Self::hampel_filter(&self.yaw_buffer, yaw, self.threshold);

        // Update buffers
        if self.pitch_buffer.len() >= self.window_size {
            self.pitch_buffer.pop_front();
        }
        if self.yaw_buffer.len() >= self.window_size {
            self.yaw_buffer.pop_front();
        }

        self.pitch_buffer.push_back(filtered_pitch);
        self.yaw_buffer.push_back(filtered_yaw);

        (filtered_pitch, filtered_yaw)
    }

    fn reset(&mut self) {
        self.pitch_buffer.clear();
        self.yaw_buffer.clear();
    }

    fn name(&self) -> &str {
        "HampelFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hampel_filter_normal() {
        let mut filter = HampelFilter::new(5, 3.0);

        // Normal values should pass through
        let values = vec![10.0, 11.0, 10.5, 11.5, 10.2];
        for val in values {
            let (p, _) = filter.apply(val, 0.0);
            assert!((p - val).abs() < 0.1);
        }
    }

    #[test]
    fn test_hampel_filter_outlier() {
        let mut filter = HampelFilter::new(5, 3.0);

        // Build up normal values
        filter.apply(10.0, 20.0);
        filter.apply(11.0, 21.0);
        filter.apply(10.5, 20.5);
        filter.apply(11.5, 21.5);

        // Apply outlier
        let (p, y) = filter.apply(100.0, 200.0);

        // Outlier should be replaced with something close to median
        assert!(p < 20.0);
        assert!(y < 30.0);
    }
}
