"""Cursor smoothing filters to reduce noise in head pose tracking."""

import numpy as np
from collections import deque


class CursorFilter:
    """Base class for cursor filters."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset filter state."""
        pass
    
    def filter(self, x, y):
        """Apply filter to x, y coordinates."""
        raise NotImplementedError


class NoFilter(CursorFilter):
    """Pass-through filter (no filtering)."""
    
    def filter(self, x, y):
        return x, y


class KalmanFilter(CursorFilter):
    """Kalman filter for 2D cursor position."""
    
    def __init__(self):
        super().__init__()
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000  # Initial uncertainty
        
        # State transition matrix
        self.F = np.array([[1, 0, 1, 0],  # x = x + vx
                          [0, 1, 0, 1],  # y = y + vy
                          [0, 0, 1, 0],  # vx = vx
                          [0, 0, 0, 1]]) # vy = vy
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],  # We measure x
                          [0, 1, 0, 0]]) # We measure y
        
        # Process noise
        self.Q = np.eye(4) * 0.1
        self.Q[2, 2] = 0.01  # Less noise in velocity
        self.Q[3, 3] = 0.01
        
        # Measurement noise
        self.R = np.eye(2) * 5
        
        self.initialized = False
    
    def reset(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.initialized = False
    
    def filter(self, x, y):
        measurement = np.array([x, y])
        
        if not self.initialized:
            self.state[0] = x
            self.state[1] = y
            self.initialized = True
            return x, y
        
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y_residual = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return int(self.state[0]), int(self.state[1])


class MovingAverageFilter(CursorFilter):
    """Moving average (FIR) filter."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        super().__init__()
    
    def reset(self):
        self.x_buffer.clear()
        self.y_buffer.clear()
    
    def filter(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        avg_x = int(np.mean(self.x_buffer))
        avg_y = int(np.mean(self.y_buffer))
        
        return avg_x, avg_y


class MedianFilter(CursorFilter):
    """Median filter for outlier rejection."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        super().__init__()
    
    def reset(self):
        self.x_buffer.clear()
        self.y_buffer.clear()
    
    def filter(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) == 0:
            return x, y
        
        med_x = int(np.median(list(self.x_buffer)))
        med_y = int(np.median(list(self.y_buffer)))
        
        return med_x, med_y


class ExponentialFilter(CursorFilter):
    """Exponential smoothing filter."""
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor (0-1, lower = smoother)
        self.last_x = None
        self.last_y = None
        super().__init__()
    
    def reset(self):
        self.last_x = None
        self.last_y = None
    
    def filter(self, x, y):
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return x, y
        
        # Exponential smoothing
        smooth_x = int(self.alpha * x + (1 - self.alpha) * self.last_x)
        smooth_y = int(self.alpha * y + (1 - self.alpha) * self.last_y)
        
        self.last_x = smooth_x
        self.last_y = smooth_y
        
        return smooth_x, smooth_y


class SecondOrderLowPassFilter(CursorFilter):
    """Second-order Butterworth low-pass filter for aggressive jitter removal."""
    
    def __init__(self, cutoff_freq=2.0, sample_rate=30.0):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        
        # Calculate filter coefficients for 2nd order Butterworth
        omega = 2 * np.pi * cutoff_freq
        omega_d = omega / sample_rate
        k = omega_d / np.tan(omega_d / 2)
        q = np.sqrt(2)  # Butterworth Q factor
        
        # Bilinear transform coefficients
        norm = k * k + k / q + 1
        self.b0 = 1 / norm
        self.b1 = 2 * self.b0
        self.b2 = self.b0
        self.a1 = 2 * (1 - k * k) / norm
        self.a2 = (k * k - k / q + 1) / norm
        
        # Initialize state variables
        self.x1 = self.x2 = 0
        self.y1 = self.y2 = 0
        self.x1_y = self.x2_y = 0
        self.y1_y = self.y2_y = 0
        self.initialized = False
        super().__init__()
    
    def reset(self):
        self.x1 = self.x2 = 0
        self.y1 = self.y2 = 0
        self.x1_y = self.x2_y = 0
        self.y1_y = self.y2_y = 0
        self.initialized = False
    
    def filter(self, x, y):
        if not self.initialized:
            # Initialize with first values
            self.x1 = self.x2 = float(x)
            self.y1 = self.y2 = float(x)
            self.x1_y = self.x2_y = float(y)
            self.y1_y = self.y2_y = float(y)
            self.initialized = True
            return x, y
        
        # Apply 2nd order filter to X coordinate
        out_x = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = out_x
        
        # Apply 2nd order filter to Y coordinate
        out_y = self.b0 * y + self.b1 * self.x1_y + self.b2 * self.x2_y - self.a1 * self.y1_y - self.a2 * self.y2_y
        self.x2_y = self.x1_y
        self.x1_y = y
        self.y2_y = self.y1_y
        self.y1_y = out_y
        
        return int(out_x), int(out_y)


class LowPassFilter(CursorFilter):
    """Simple RC low-pass filter for removing high-frequency jitter."""
    
    def __init__(self, cutoff_freq=5.0, sample_rate=30.0):
        self.cutoff_freq = cutoff_freq  # Cutoff frequency in Hz
        self.sample_rate = sample_rate  # Sample rate in Hz
        
        # Calculate RC time constant
        rc = 1.0 / (2.0 * np.pi * cutoff_freq)
        dt = 1.0 / sample_rate
        self.alpha = dt / (rc + dt)
        
        # Initialize state
        self.last_x = None
        self.last_y = None
        super().__init__()
    
    def reset(self):
        self.last_x = None
        self.last_y = None
    
    def filter(self, x, y):
        if self.last_x is None:
            self.last_x = float(x)
            self.last_y = float(y)
            return x, y
        
        # Apply RC low-pass filter
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        self.last_x = self.alpha * x + (1 - self.alpha) * self.last_x
        self.last_y = self.alpha * y + (1 - self.alpha) * self.last_y
        
        return int(self.last_x), int(self.last_y)


class HampelFilter(CursorFilter):
    """Hampel filter for robust outlier detection and smoothing."""
    
    def __init__(self, window_size=7, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold  # Number of median absolute deviations
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        super().__init__()
    
    def reset(self):
        self.x_buffer.clear()
        self.y_buffer.clear()
    
    def filter(self, x, y):
        # Add new values to buffers
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        # Need at least 3 samples for Hampel filter
        if len(self.x_buffer) < 3:
            return x, y
        
        # Apply Hampel filter to X coordinate
        x_values = np.array(list(self.x_buffer))
        x_median = np.median(x_values)
        x_mad = np.median(np.abs(x_values - x_median))
        
        # Robust standard deviation estimate
        x_sigma = 1.4826 * x_mad
        
        # Check if current value is an outlier
        if x_sigma > 0 and np.abs(x - x_median) > self.threshold * x_sigma:
            # Replace outlier with median
            filtered_x = x_median
        else:
            # Use exponential smoothing for non-outliers
            if len(self.x_buffer) > 1:
                filtered_x = 0.7 * x + 0.3 * x_values[-2]
            else:
                filtered_x = x
        
        # Apply Hampel filter to Y coordinate
        y_values = np.array(list(self.y_buffer))
        y_median = np.median(y_values)
        y_mad = np.median(np.abs(y_values - y_median))
        
        # Robust standard deviation estimate
        y_sigma = 1.4826 * y_mad
        
        # Check if current value is an outlier
        if y_sigma > 0 and np.abs(y - y_median) > self.threshold * y_sigma:
            # Replace outlier with median
            filtered_y = y_median
        else:
            # Use exponential smoothing for non-outliers
            if len(self.y_buffer) > 1:
                filtered_y = 0.7 * y + 0.3 * y_values[-2]
            else:
                filtered_y = y
        
        return int(filtered_x), int(filtered_y)


def create_cursor_filter(filter_type):
    """Factory function to create cursor filters."""
    filter_map = {
        'none': NoFilter,
        'kalman': KalmanFilter,
        'moving_average': lambda: MovingAverageFilter(window_size=5),
        'fir': lambda: MovingAverageFilter(window_size=5),
        'median': lambda: MedianFilter(window_size=5),
        'exponential': lambda: ExponentialFilter(alpha=0.3),
        'exp': lambda: ExponentialFilter(alpha=0.3),
        'lowpass': lambda: LowPassFilter(cutoff_freq=2.0),
        'low_pass': lambda: LowPassFilter(cutoff_freq=2.0),
        'lowpass2': lambda: SecondOrderLowPassFilter(cutoff_freq=2.0),
        'low_pass2': lambda: SecondOrderLowPassFilter(cutoff_freq=2.0),
        'hampel': lambda: HampelFilter(window_size=7, threshold=3.0),
    }
    
    filter_type = filter_type.lower()
    if filter_type not in filter_map:
        raise ValueError(f"Unknown filter type: {filter_type}. Available: {list(filter_map.keys())}")
    
    filter_class = filter_map[filter_type]
    return filter_class() if callable(filter_class) else filter_class