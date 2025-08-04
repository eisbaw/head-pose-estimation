"""Movement detection for head pose tracking."""

import numpy as np
from collections import deque


class MovementDetector:
    """Detect if head is moving or still based on pose history."""
    
    def __init__(self, window_size=15, std_threshold=2.0, range_threshold=5.0):
        """
        Initialize movement detector.
        
        Args:
            window_size: Number of frames to analyze
            std_threshold: Standard deviation threshold in degrees
            range_threshold: Range (max-min) threshold in degrees
        """
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.range_threshold = range_threshold
        
        self.pitch_buffer = deque(maxlen=window_size)
        self.yaw_buffer = deque(maxlen=window_size)
        
    def reset(self):
        """Reset the detector state."""
        self.pitch_buffer.clear()
        self.yaw_buffer.clear()
        
    def update(self, pitch, yaw):
        """
        Update with new pose data and determine if moving.
        
        Args:
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            
        Returns:
            bool: True if moving, False if still
        """
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        
        # Need enough samples
        if len(self.pitch_buffer) < 5:
            return False  # Assume still until we have data
        
        # Calculate statistics
        pitch_array = np.array(self.pitch_buffer)
        yaw_array = np.array(self.yaw_buffer)
        
        pitch_std = np.std(pitch_array)
        yaw_std = np.std(yaw_array)
        
        pitch_range = np.max(pitch_array) - np.min(pitch_array)
        yaw_range = np.max(yaw_array) - np.min(yaw_array)
        
        # Decision tree
        # 1. Check if standard deviation exceeds threshold
        if pitch_std > self.std_threshold or yaw_std > self.std_threshold:
            return True
            
        # 2. Check if range exceeds threshold
        if pitch_range > self.range_threshold or yaw_range > self.range_threshold:
            return True
            
        # 3. Check for rapid changes (optional - for very quick movements)
        if len(self.pitch_buffer) >= 2:
            recent_pitch_change = abs(pitch_array[-1] - pitch_array[-2])
            recent_yaw_change = abs(yaw_array[-1] - yaw_array[-2])
            if recent_pitch_change > 1.0 or recent_yaw_change > 1.0:
                return True
        
        # If none of the conditions are met, head is still
        return False
    
    def get_stats(self):
        """Get current statistics for debugging."""
        if len(self.pitch_buffer) < 2:
            return None
            
        pitch_array = np.array(self.pitch_buffer)
        yaw_array = np.array(self.yaw_buffer)
        
        return {
            'pitch_std': np.std(pitch_array),
            'yaw_std': np.std(yaw_array),
            'pitch_range': np.max(pitch_array) - np.min(pitch_array),
            'yaw_range': np.max(yaw_array) - np.min(yaw_array),
            'samples': len(self.pitch_buffer)
        }