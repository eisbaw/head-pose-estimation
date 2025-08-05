#!/usr/bin/env python3
"""Generate reference values for filter tests."""

import sys
sys.path.append('.')

from cursor_filters import (
    MovingAverageFilter,
    ExponentialFilter,
    MedianFilter,
    LowPassFilter,
    SecondOrderLowPassFilter,
    HampelFilter
)

# Test sequence
test_sequence = [
    (10.0, 20.0),
    (15.0, 25.0),
    (20.0, 30.0),
    (25.0, 35.0),
    (30.0, 40.0),
    (35.0, 45.0),
    (40.0, 50.0),
    (45.0, 55.0),
    (50.0, 60.0),
    (55.0, 65.0),
]

def test_filter(filter_obj, name):
    print(f"\n{name}:")
    print("Input -> Output")
    results = []
    for x, y in test_sequence:
        out_x, out_y = filter_obj.filter(x, y)
        print(f"({x}, {y}) -> ({out_x}, {out_y})")
        results.append((float(out_x), float(out_y)))
    
    # Print as Rust array
    print(f"\n/// Expected outputs from Python {name}")
    print(f"pub const {name.upper().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_').replace('.', '_')}_EXPECTED: [(f64, f64); 10] = [")
    for i, (x, y) in enumerate(results):
        print(f"    ({x}, {y}),   // {test_sequence[i]}")
    print("];")

print("=== Filter Reference Values ===")

# Test each filter
test_filter(MovingAverageFilter(window_size=3), "MovingAverageFilter(window_size=3)")
test_filter(ExponentialFilter(alpha=0.3), "ExponentialFilter(alpha=0.3)")
test_filter(MedianFilter(window_size=3), "MedianFilter(window_size=3)")
test_filter(LowPassFilter(cutoff_freq=5.0, sample_rate=30.0), "LowPassFilter(cutoff_freq=5.0)")

# Also test the actual alpha value for low pass
lp = LowPassFilter(cutoff_freq=5.0, sample_rate=30.0)
print(f"\nLowPass filter alpha = {lp.alpha}")

# Test Hampel with outlier
print("\n=== Hampel Filter Outlier Test ===")
hampel = HampelFilter(window_size=5, threshold=3.0)
normal_values = [(10.0, 20.0), (11.0, 21.0), (12.0, 22.0), (13.0, 23.0)]
for x, y in normal_values:
    out_x, out_y = hampel.filter(x, y)
    print(f"Normal: ({x}, {y}) -> ({out_x}, {out_y})")

# Outlier
out_x, out_y = hampel.filter(100.0, 200.0)
print(f"Outlier: (100.0, 200.0) -> ({out_x}, {out_y})")