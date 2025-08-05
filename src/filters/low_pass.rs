use super::CursorFilter;
use crate::constants::{DEFAULT_FPS, OMEGA_FACTOR};

/// First-order low-pass filter
pub struct LowPassFilter {
    alpha: f64,
    last_pitch: Option<f64>,
    last_yaw: Option<f64>,
}

impl LowPassFilter {
    /// Create a new first-order low-pass filter
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in the range (0, 1]
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "Alpha must be in (0, 1]");
        Self {
            alpha,
            last_pitch: None,
            last_yaw: None,
        }
    }
}

impl CursorFilter for LowPassFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        let filtered_pitch = match self.last_pitch {
            Some(last) => self.alpha.mul_add(pitch - last, last),
            None => pitch,
        };

        let filtered_yaw = match self.last_yaw {
            Some(last) => self.alpha.mul_add(yaw - last, last),
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
        "LowPassFilter"
    }
}

/// Second-order low-pass filter
pub struct SecondOrderLowPassFilter {
    cutoff_freq: f64,
    damping: f64,
    dt: f64,

    // State variables for pitch
    pitch_x: f64,
    pitch_dx: f64,
    pitch_ddx: f64,

    // State variables for yaw
    yaw_x: f64,
    yaw_dx: f64,
    yaw_ddx: f64,

    initialized: bool,
}

impl SecondOrderLowPassFilter {
    /// Create a new second-order low-pass filter
    #[must_use]
    pub fn new(cutoff_freq: f64, damping: f64) -> Self {
        let dt = 1.0 / DEFAULT_FPS;

        Self {
            cutoff_freq,
            damping,
            dt,
            pitch_x: 0.0,
            pitch_dx: 0.0,
            pitch_ddx: 0.0,
            yaw_x: 0.0,
            yaw_dx: 0.0,
            yaw_ddx: 0.0,
            initialized: false,
        }
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn update_state(x: &mut f64, dx: &mut f64, ddx: &mut f64, input: f64, omega: f64, zeta: f64, dt: f64) -> f64 {
        // Calculate acceleration
        // omega² * (input - position) - 2 * zeta * omega * velocity
        *ddx = (omega * omega).mul_add(input - *x, -(2.0 * zeta * omega * *dx));

        // Update velocity and position
        *dx += *ddx * dt;
        *x += *dx * dt;

        *x
    }
}

impl CursorFilter for SecondOrderLowPassFilter {
    fn apply(&mut self, pitch: f64, yaw: f64) -> (f64, f64) {
        let omega = OMEGA_FACTOR * std::f64::consts::PI * self.cutoff_freq;

        if !self.initialized {
            self.pitch_x = pitch;
            self.yaw_x = yaw;
            self.initialized = true;
        }

        let filtered_pitch = Self::update_state(
            &mut self.pitch_x,
            &mut self.pitch_dx,
            &mut self.pitch_ddx,
            pitch,
            omega,
            self.damping,
            self.dt,
        );

        let filtered_yaw = Self::update_state(
            &mut self.yaw_x,
            &mut self.yaw_dx,
            &mut self.yaw_ddx,
            yaw,
            omega,
            self.damping,
            self.dt,
        );

        (filtered_pitch, filtered_yaw)
    }

    fn reset(&mut self) {
        self.pitch_x = 0.0;
        self.pitch_dx = 0.0;
        self.pitch_ddx = 0.0;
        self.yaw_x = 0.0;
        self.yaw_dx = 0.0;
        self.yaw_ddx = 0.0;
        self.initialized = false;
    }

    fn name(&self) -> &str {
        "SecondOrderLowPassFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_order_low_pass() {
        let mut filter = LowPassFilter::new(0.5);

        // First value passes through
        let (p1, y1) = filter.apply(10.0, 20.0);
        assert_eq!(p1, 10.0);
        assert_eq!(y1, 20.0);

        // Second value is filtered
        let (p2, y2) = filter.apply(20.0, 30.0);
        assert_eq!(p2, 15.0); // 10 + 0.5 * (20 - 10)
        assert_eq!(y2, 25.0);
    }

    #[test]
    fn test_second_order_low_pass() {
        let mut filter = SecondOrderLowPassFilter::new(5.0, 0.707);

        // Apply step input
        let mut last_pitch = 0.0;
        for _ in 0..30 {
            let (p, _) = filter.apply(10.0, 0.0);
            // Should gradually approach target
            assert!(p >= last_pitch);
            assert!(p <= 10.0);
            last_pitch = p;
        }

        // Should be close to target after settling
        assert!((last_pitch - 10.0).abs() < 0.1);
    }
}
