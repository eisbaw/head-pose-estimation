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
    }
    
    filter_type = filter_type.lower()
    if filter_type not in filter_map:
        raise ValueError(f"Unknown filter type: {filter_type}. Available: {list(filter_map.keys())}")
    
    filter_class = filter_map[filter_type]
    return filter_class() if callable(filter_class) else filter_class