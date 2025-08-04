use super::CursorFilter;

/// Exponential smoothing filter
pub struct ExponentialFilter {
    alpha: f64,
    last_pitch: Option<f64>,
    last_yaw: Option<f64>,
}

impl ExponentialFilter {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "Alpha must be in (0, 1]");
        Self {
            alpha,
            last_pitch: None,
            last_yaw: None,
        }
    }
}

impl CursorFilter for ExponentialFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        let filtered_pitch = match self.last_pitch {
            Some(last) => self.alpha * pitch + (1.0 - self.alpha) * last,
            None => pitch,
        };
        
        let filtered_yaw = match self.last_yaw {
            Some(last) => self.alpha * yaw + (1.0 - self.alpha) * last,
            None => yaw,
        };
        
        self.last_pitch = Some(filtered_pitch);
        self.last_yaw = Some(filtered_yaw);
        
        (filtered_pitch, filtered_yaw)
    }
    
    fn reset(&mut self) {
        self.last_pitch = None;
        self.last_yaw = None;
    }
    
    fn name(&self) -> &str {
        "ExponentialFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_filter() {
        let mut filter = ExponentialFilter::new(0.5);
        
        // First value passes through
        let (p1, y1) = filter.apply(10.0, 20.0);
        assert_eq!(p1, 10.0);
        assert_eq!(y1, 20.0);
        
        // Second value is smoothed
        let (p2, y2) = filter.apply(20.0, 30.0);
        assert_eq!(p2, 15.0); // 0.5 * 20 + 0.5 * 10
        assert_eq!(y2, 25.0);
    }
    
    #[test]
    fn test_alpha_bounds() {
        // High alpha = less smoothing
        let mut filter1 = ExponentialFilter::new(0.9);
        filter1.apply(10.0, 20.0);
        let (p, _y) = filter1.apply(20.0, 30.0);
        assert!((p - 19.0).abs() < 0.001); // 0.9 * 20 + 0.1 * 10
        
        // Low alpha = more smoothing
        let mut filter2 = ExponentialFilter::new(0.1);
        filter2.apply(10.0, 20.0);
        let (p, _y) = filter2.apply(20.0, 30.0);
        assert!((p - 11.0).abs() < 0.001); // 0.1 * 20 + 0.9 * 10
    }
}