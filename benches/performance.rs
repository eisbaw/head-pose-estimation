//! Performance benchmarks for head pose estimation components

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use head_pose_estimation::{
    filters::create_filter,
    movement_detector::MovementDetector,
};
use opencv::{core::Mat, prelude::*};
use std::time::Duration;

/// Benchmark different filter implementations
fn bench_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("filters");
    group.measurement_time(Duration::from_secs(10));
    
    let filter_names = vec![
        "none",
        "moving_average",
        "median", 
        "exponential",
        "kalman",
        "lowpass",
        "lowpass2",
        "hampel",
    ];
    
    // Test data: simulate noisy sensor readings
    let test_data: Vec<(f64, f64)> = (0..1000)
        .map(|i| {
            let t = i as f64 * 0.01;
            let true_pitch = (t * 2.0).sin() * 30.0;
            let true_yaw = (t * 1.5).cos() * 25.0;
            let noise_pitch = ((i * 17) % 11) as f64 / 11.0 - 0.5;
            let noise_yaw = ((i * 13) % 7) as f64 / 7.0 - 0.5;
            (true_pitch + noise_pitch * 5.0, true_yaw + noise_yaw * 5.0)
        })
        .collect();
    
    for filter_name in filter_names {
        group.bench_with_input(
            BenchmarkId::new("apply", filter_name),
            &filter_name,
            |b, &name| {
                let mut filter = create_filter(name).unwrap();
                let data = test_data.clone();
                b.iter(|| {
                    for &(pitch, yaw) in &data {
                        let _ = black_box(filter.apply(pitch, yaw));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark filter reset operations
fn bench_filter_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_reset");
    
    let filter_names = vec![
        "moving_average",
        "median",
        "exponential", 
        "kalman",
        "lowpass",
        "hampel",
    ];
    
    for filter_name in filter_names {
        group.bench_with_input(
            BenchmarkId::new("reset", filter_name),
            &filter_name,
            |b, &name| {
                let mut filter = create_filter(name).unwrap();
                // Pre-fill the filter with some data
                for i in 0..100 {
                    filter.apply(i as f64, i as f64);
                }
                b.iter(|| {
                    filter.reset();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark movement detection
fn bench_movement_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("movement_detection");
    
    let window_sizes = vec![10, 30, 60, 120];
    
    for window_size in window_sizes {
        group.bench_with_input(
            BenchmarkId::new("update", window_size),
            &window_size,
            |b, &size| {
                let mut detector = MovementDetector::new(size, 1.0);
                let test_data: Vec<(f64, f64)> = (0..size * 2)
                    .map(|i| {
                        let angle = i as f64 * 0.1;
                        (angle.sin() * 20.0, angle.cos() * 20.0)
                    })
                    .collect();
                
                b.iter(|| {
                    for &(pitch, yaw) in &test_data {
                        let _ = black_box(detector.update(pitch, yaw));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark OpenCV operations
fn bench_opencv_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("opencv");
    
    // Benchmark Mat creation
    group.bench_function("mat_creation_640x480", |b| {
        b.iter(|| {
            let mat = Mat::zeros(480, 640, opencv::core::CV_8UC3)
                .unwrap()
                .to_mat()
                .unwrap();
            black_box(mat);
        });
    });
    
    // Benchmark ROI extraction
    let image = Mat::zeros(480, 640, opencv::core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();
    let rect = opencv::core::Rect::new(100, 100, 200, 200);
    
    group.bench_function("roi_extraction", |b| {
        b.iter(|| {
            let roi = Mat::roi(&image, rect).unwrap();
            black_box(roi);
        });
    });
    
    group.finish();
}

/// Benchmark filter comparisons for algorithm selection
fn bench_filter_accuracy_vs_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_tradeoffs");
    group.measurement_time(Duration::from_secs(20));
    
    // Generate test sequence with known characteristics
    let clean_signal: Vec<(f64, f64)> = (0..10000)
        .map(|i| {
            let t = i as f64 * 0.001;
            ((t * 3.0).sin() * 30.0, (t * 2.0).cos() * 25.0)
        })
        .collect();
    
    // Add different types of noise
    let noisy_signal: Vec<(f64, f64)> = clean_signal.iter()
        .enumerate()
        .map(|(i, &(pitch, yaw))| {
            let mut rng = i as u32;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let gaussian_noise = ((rng / 65536) % 1000) as f64 / 1000.0 - 0.5;
            
            // Add occasional spikes
            let spike = if i % 100 == 50 { 20.0 } else { 0.0 };
            
            (
                pitch + gaussian_noise * 3.0 + spike,
                yaw + gaussian_noise * 3.0 - spike
            )
        })
        .collect();
    
    let filters = vec![
        ("moving_average", create_filter("moving_average").unwrap()),
        ("median", create_filter("median").unwrap()),
        ("kalman", create_filter("kalman").unwrap()),
        ("lowpass", create_filter("lowpass").unwrap()),
        ("hampel", create_filter("hampel").unwrap()),
    ];
    
    for (name, mut filter) in filters {
        group.bench_with_input(
            BenchmarkId::new("process_10k_samples", name),
            &name,
            |b, _| {
                b.iter(|| {
                    filter.reset();
                    for &(noisy_pitch, noisy_yaw) in &noisy_signal {
                        let (filtered_pitch, filtered_yaw) = filter.apply(noisy_pitch, noisy_yaw);
                        black_box((filtered_pitch, filtered_yaw));
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_filters,
    bench_filter_reset,
    bench_movement_detection,
    bench_opencv_operations,
    bench_filter_accuracy_vs_speed
);
criterion_main!(benches);