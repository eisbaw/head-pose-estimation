//! Benchmarks for filter performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use head_pose_estimation::filters::{
    exponential::ExponentialFilter,
    hampel::HampelFilter,
    kalman::KalmanFilter,
    low_pass::{LowPassFilter, SecondOrderLowPassFilter},
    median::MedianFilter,
    moving_average::MovingAverageFilter,
    CursorFilter, NoFilter,
};

fn benchmark_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("filters");

    // Test data - simulating noisy head pose measurements
    let test_data: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let t = i as f64 * 0.1;
            let pitch = 10.0 * t.sin() + 0.5 * rand::random::<f64>();
            let yaw = 15.0 * t.cos() + 0.5 * rand::random::<f64>();
            (pitch, yaw)
        })
        .collect();

    // Benchmark each filter type
    let filter_configs = vec![
        ("no_filter", Box::new(NoFilter) as Box<dyn CursorFilter>),
        ("moving_average_5", Box::new(MovingAverageFilter::new(5))),
        ("moving_average_10", Box::new(MovingAverageFilter::new(10))),
        ("median_5", Box::new(MedianFilter::new(5))),
        ("median_9", Box::new(MedianFilter::new(9))),
        ("exponential_0.5", Box::new(ExponentialFilter::new(0.5))),
        ("exponential_0.8", Box::new(ExponentialFilter::new(0.8))),
        ("kalman", Box::new(KalmanFilter::new())),
        ("low_pass_0.3", Box::new(LowPassFilter::new(0.3))),
        (
            "second_order_low_pass",
            Box::new(SecondOrderLowPassFilter::new(10.0, 0.7)),
        ),
        ("hampel_7", Box::new(HampelFilter::new(7, 3.0))),
    ];

    for (name, mut filter) in filter_configs {
        group.bench_with_input(
            BenchmarkId::new("single_update", name),
            &test_data[0],
            |b, &(pitch, yaw)| {
                b.iter(|| black_box(filter.apply(black_box(pitch), black_box(yaw))));
            },
        );

        group.bench_with_input(BenchmarkId::new("sequence_100", name), &test_data, |b, data| {
            b.iter(|| {
                filter.reset();
                for &(pitch, yaw) in data {
                    black_box(filter.apply(black_box(pitch), black_box(yaw)));
                }
            });
        });
    }

    group.finish();
}

fn benchmark_filter_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_reset");

    let filter_configs = vec![
        (
            "moving_average",
            Box::new(MovingAverageFilter::new(10)) as Box<dyn CursorFilter>,
        ),
        ("median", Box::new(MedianFilter::new(9))),
        ("kalman", Box::new(KalmanFilter::new())),
        ("hampel", Box::new(HampelFilter::new(7, 3.0))),
    ];

    for (name, mut filter) in filter_configs {
        // Pre-fill filter with data
        for i in 0..20 {
            filter.apply(i as f64, i as f64);
        }

        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(filter.reset());
            });
        });
    }

    group.finish();
}

fn benchmark_kalman_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kalman_matrix_ops");

    let mut kalman = KalmanFilter::new();

    // Benchmark the most expensive operations in Kalman filter
    group.bench_function("kalman_predict_update_cycle", |b| {
        b.iter(|| {
            kalman.reset();
            for i in 0..10 {
                let val = i as f64;
                black_box(kalman.apply(black_box(val), black_box(val)));
            }
        });
    });

    group.finish();
}

fn benchmark_median_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("median_sorting");

    for window_size in [5, 9, 15, 21] {
        let mut median = MedianFilter::new(window_size);

        group.bench_with_input(BenchmarkId::new("window_size", window_size), &window_size, |b, _| {
            b.iter(|| {
                median.reset();
                for i in 0..window_size {
                    black_box(median.apply(black_box(i as f64), black_box(i as f64)));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_filters,
    benchmark_filter_reset,
    benchmark_kalman_matrix_operations,
    benchmark_median_sorting
);
criterion_main!(benches);
