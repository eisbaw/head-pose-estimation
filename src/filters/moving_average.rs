use super::CursorFilter;
use std::collections::VecDeque;

/// Moving average filter
pub struct MovingAverageFilter {
    window_size: usize,
    pitch_buffer: VecDeque<f64>,
    yaw_buffer: VecDeque<f64>,
}

impl MovingAverageFilter {
    /// Create a new moving average filter
    ///
    /// # Panics
    ///
    /// Panics if window_size is 0
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be greater than 0");
        Self {
            window_size,
            pitch_buffer: VecDeque::with_capacity(window_size),
            yaw_buffer: VecDeque::with_capacity(window_size),
        }
    }
}

impl CursorFilter for MovingAverageFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        // Add to buffers
        if self.pitch_buffer.len() >= self.window_size {
            self.pitch_buffer.pop_front();
        }
        if self.yaw_buffer.len() >= self.window_size {
            self.yaw_buffer.pop_front();
        }

        self.pitch_buffer.push_back(pitch);
        self.yaw_buffer.push_back(yaw);

        // Calculate averages
        #[allow(clippy::cast_precision_loss)] // Buffer size is small
        let pitch_avg = self.pitch_buffer.iter().sum::<f64>() / self.pitch_buffer.len() as f64;
        #[allow(clippy::cast_precision_loss)] // Buffer size is small
        let yaw_avg = self.yaw_buffer.iter().sum::<f64>() / self.yaw_buffer.len() as f64;

        (pitch_avg, yaw_avg)
    }

    fn reset(&mut self) {
        self.pitch_buffer.clear();
        self.yaw_buffer.clear();
    }

    fn name(&self) -> &str {
        "MovingAverageFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let mut filter = MovingAverageFilter::new(3);

        let (p1, y1) = filter.apply(10.0, 20.0);
        assert_eq!(p1, 10.0);
        assert_eq!(y1, 20.0);

        let (p2, y2) = filter.apply(20.0, 30.0);
        assert_eq!(p2, 15.0);
        assert_eq!(y2, 25.0);

        let (p3, y3) = filter.apply(30.0, 40.0);
        assert_eq!(p3, 20.0);
        assert_eq!(y3, 30.0);

        // Window is full, oldest value should be dropped
        let (p4, y4) = filter.apply(40.0, 50.0);
        assert_eq!(p4, 30.0);
        assert_eq!(y4, 40.0);
    }
}
