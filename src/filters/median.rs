use std::collections::VecDeque;
use super::CursorFilter;

/// Median filter
pub struct MedianFilter {
    window_size: usize,
    pitch_buffer: VecDeque<f64>,
    yaw_buffer: VecDeque<f64>,
}

impl MedianFilter {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            pitch_buffer: VecDeque::with_capacity(window_size),
            yaw_buffer: VecDeque::with_capacity(window_size),
        }
    }
    
    fn calculate_median(values: &VecDeque<f64>) -> f64 {
        let mut sorted: Vec<f64> = values.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len == 0 {
            0.0
        } else if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
}

impl CursorFilter for MedianFilter {
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
        
        // Calculate medians
        let pitch_median = Self::calculate_median(&self.pitch_buffer);
        let yaw_median = Self::calculate_median(&self.yaw_buffer);
        
        (pitch_median, yaw_median)
    }
    
    fn reset(&mut self) {
        self.pitch_buffer.clear();
        self.yaw_buffer.clear();
    }
    
    fn name(&self) -> &str {
        "MedianFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_filter() {
        let mut filter = MedianFilter::new(3);
        
        let (p1, y1) = filter.apply(10.0, 20.0);
        assert_eq!(p1, 10.0);
        assert_eq!(y1, 20.0);
        
        let (p2, y2) = filter.apply(20.0, 30.0);
        assert_eq!(p2, 15.0); // median of [10, 20]
        assert_eq!(y2, 25.0);
        
        let (p3, y3) = filter.apply(30.0, 40.0);
        assert_eq!(p3, 20.0); // median of [10, 20, 30]
        assert_eq!(y3, 30.0);
    }
    
    #[test]
    fn test_median_with_outliers() {
        let mut filter = MedianFilter::new(3);
        
        filter.apply(10.0, 20.0);
        filter.apply(11.0, 21.0);
        let (p, y) = filter.apply(100.0, 200.0); // outlier
        
        // Median should filter out the outlier
        assert_eq!(p, 11.0);
        assert_eq!(y, 21.0);
    }
}