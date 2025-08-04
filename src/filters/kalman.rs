use nalgebra::{Matrix2, Matrix4, Vector2, Vector4};
use super::CursorFilter;

/// Kalman filter for smooth cursor movement
pub struct KalmanFilter {
    // State: [x, y, vx, vy]
    state: Vector4<f64>,
    // State covariance
    covariance: Matrix4<f64>,
    // Process noise
    process_noise: Matrix4<f64>,
    // Measurement noise
    measurement_noise: Matrix2<f64>,
    // State transition matrix
    transition: Matrix4<f64>,
    // Measurement matrix
    measurement: Matrix2x4<f64>,
    // Time step
    dt: f64,
}

type Matrix2x4<T> = nalgebra::Matrix<T, nalgebra::U2, nalgebra::U4, nalgebra::ArrayStorage<T, 2, 4>>;

impl KalmanFilter {
    pub fn new() -> Self {
        let dt: f64 = 1.0 / 30.0; // Assume 30 FPS
        
        // State transition matrix
        let transition = Matrix4::new(
            1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        
        // Measurement matrix (we only measure position)
        let measurement = Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );
        
        // Process noise
        let q: f64 = 0.1;
        let process_noise = Matrix4::new(
            q * dt.powi(4) / 4.0, 0.0, q * dt.powi(3) / 2.0, 0.0,
            0.0, q * dt.powi(4) / 4.0, 0.0, q * dt.powi(3) / 2.0,
            q * dt.powi(3) / 2.0, 0.0, q * dt.powi(2), 0.0,
            0.0, q * dt.powi(3) / 2.0, 0.0, q * dt.powi(2),
        );
        
        // Measurement noise
        let r = 1.0;
        let measurement_noise = Matrix2::new(
            r, 0.0,
            0.0, r,
        );
        
        // Initial state and covariance
        let state = Vector4::zeros();
        let covariance = Matrix4::identity() * 1000.0;
        
        Self {
            state,
            covariance,
            process_noise,
            measurement_noise,
            transition,
            measurement,
            dt,
        }
    }
    
    fn predict(&mut self) {
        // Predict state
        self.state = self.transition * self.state;
        
        // Predict covariance
        self.covariance = self.transition * self.covariance * self.transition.transpose() + self.process_noise;
    }
    
    fn update(&mut self, measurement: Vector2<f64>) {
        // Innovation
        let innovation = measurement - self.measurement * self.state;
        
        // Innovation covariance
        let innovation_cov = self.measurement * self.covariance * self.measurement.transpose() + self.measurement_noise;
        
        // Kalman gain
        let gain = self.covariance * self.measurement.transpose() * innovation_cov.try_inverse().unwrap();
        
        // Update state
        self.state += gain * innovation;
        
        // Update covariance
        let identity = Matrix4::identity();
        self.covariance = (identity - gain * self.measurement) * self.covariance;
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl CursorFilter for KalmanFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        // Predict
        self.predict();
        
        // Update with measurement
        let measurement = Vector2::new(pitch, yaw);
        self.update(measurement);
        
        // Return filtered position
        (self.state[0], self.state[1])
    }
    
    fn reset(&mut self) {
        self.state = Vector4::zeros();
        self.covariance = Matrix4::identity() * 1000.0;
    }
    
    fn name(&self) -> &str {
        "KalmanFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter() {
        let mut filter = KalmanFilter::new();
        
        // First measurement initializes the filter
        let (p1, y1) = filter.apply(10.0, 20.0);
        assert!((p1 - 10.0).abs() < 1.0);
        assert!((y1 - 20.0).abs() < 1.0);
        
        // Subsequent measurements should be smoothed
        let (p2, y2) = filter.apply(11.0, 21.0);
        assert!(p2 > 10.0 && p2 < 11.0);
        assert!(y2 > 20.0 && y2 < 21.0);
    }
}