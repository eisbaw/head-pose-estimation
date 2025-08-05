//! Benchmarks comparing filter accuracy and performance characteristics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use head_pose_estimation::filters::{
    exponential::ExponentialFilter,
    hampel::HampelFilter,
    kalman::KalmanFilter,
    low_pass::{LowPassFilter, SecondOrderLowPassFilter},
    median::MedianFilter,
    moving_average::MovingAverageFilter,
    CursorFilter, NoFilter,
};
use std::f64::consts::PI;

/// Generate test signal with noise
fn generate_test_signal(num_samples: usize) -> Vec<(f64, f64, f64, f64)> {
    (0..num_samples)
        .map(|i| {
            let t = i as f64 * 0.01; // 100 Hz sampling

            // True signal: smooth sinusoidal head movement
            let true_pitch = 15.0 * (2.0 * PI * 0.5 * t).sin();
            let true_yaw = 20.0 * (2.0 * PI * 0.3 * t).cos();

            // Add Gaussian noise
            let noise_level = 2.0;
            let pitch_noise = noise_level * (rand::random::<f64>() - 0.5);
            let yaw_noise = noise_level * (rand::random::<f64>() - 0.5);

            // Occasionally add outliers (5% chance)
            let (pitch_noise, yaw_noise) = if rand::random::<f64>() < 0.05 {
                (pitch_noise * 10.0, yaw_noise * 10.0)
            } else {
                (pitch_noise, yaw_noise)
            };

            let noisy_pitch = true_pitch + pitch_noise;
            let noisy_yaw = true_yaw + yaw_noise;

            (true_pitch, true_yaw, noisy_pitch, noisy_yaw)
        })
        .collect()
}

/// Calculate root mean square error
fn calculate_rmse(filtered: &[(f64, f64)], truth: &[(f64, f64)]) -> f64 {
    let sum_squared_error: f64 = filtered
        .iter()
        .zip(truth.iter())
        .map(|((fp, fy), (tp, ty))| {
            let pitch_error = fp - tp;
            let yaw_error = fy - ty;
            pitch_error * pitch_error + yaw_error * yaw_error
        })
        .sum();

    (sum_squared_error / filtered.len() as f64).sqrt()
}

/// Calculate lag (phase shift) between filtered and true signal
fn calculate_lag(filtered: &[(f64, f64)], truth: &[(f64, f64)]) -> usize {
    // Simple cross-correlation to find lag
    let max_lag = 50; // Check up to 50 samples lag
    let mut best_lag = 0;
    let mut best_correlation = f64::NEG_INFINITY;

    for lag in 0..max_lag.min(filtered.len() / 2) {
        let correlation: f64 = filtered[lag..]
            .iter()
            .zip(truth.iter())
            .take(filtered.len() - lag)
            .map(|((fp, _), (tp, _))| fp * tp)
            .sum();

        if correlation > best_correlation {
            best_correlation = correlation;
            best_lag = lag;
        }
    }

    best_lag
}

fn benchmark_filter_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_accuracy");
    group.plot_config(PlotConfiguration::default());

    let test_signal = generate_test_signal(1000);
    let truth: Vec<(f64, f64)> = test_signal.iter().map(|(tp, ty, _, _)| (*tp, *ty)).collect();
    let noisy: Vec<(f64, f64)> = test_signal.iter().map(|(_, _, np, ny)| (*np, *ny)).collect();

    let filter_configs: Vec<(&str, Box<dyn CursorFilter>)> = vec![
        ("no_filter", Box::new(NoFilter)),
        ("moving_avg_5", Box::new(MovingAverageFilter::new(5))),
        ("moving_avg_10", Box::new(MovingAverageFilter::new(10))),
        ("median_5", Box::new(MedianFilter::new(5))),
        ("median_9", Box::new(MedianFilter::new(9))),
        ("exponential_0.3", Box::new(ExponentialFilter::new(0.3))),
        ("exponential_0.7", Box::new(ExponentialFilter::new(0.7))),
        ("kalman", Box::new(KalmanFilter::new())),
        ("low_pass_0.3", Box::new(LowPassFilter::new(0.3))),
        ("low_pass_0.7", Box::new(LowPassFilter::new(0.7))),
        ("second_order_10hz", Box::new(SecondOrderLowPassFilter::new(10.0, 0.7))),
        ("hampel_7", Box::new(HampelFilter::new(7, 3.0))),
    ];

    for (name, mut filter) in filter_configs {
        group.bench_with_input(
            BenchmarkId::new("process_and_measure", name),
            &noisy,
            |b, noisy_data| {
                b.iter(|| {
                    filter.reset();
                    let mut filtered = Vec::with_capacity(noisy_data.len());

                    for &(pitch, yaw) in noisy_data {
                        let (fp, fy) = filter.apply(black_box(pitch), black_box(yaw));
                        filtered.push((fp, fy));
                    }

                    // Calculate metrics
                    let rmse = calculate_rmse(&filtered, &truth);
                    let lag = calculate_lag(&filtered, &truth);

                    black_box((filtered, rmse, lag))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_filter_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_convergence");

    // Test how quickly filters converge to a step input
    let step_value = 45.0;
    let num_samples = 50;

    let filter_configs: Vec<(&str, Box<dyn CursorFilter>)> = vec![
        ("moving_avg_5", Box::new(MovingAverageFilter::new(5))),
        ("exponential_0.5", Box::new(ExponentialFilter::new(0.5))),
        ("kalman", Box::new(KalmanFilter::new())),
        ("low_pass_0.5", Box::new(LowPassFilter::new(0.5))),
    ];

    for (name, mut filter) in filter_configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                filter.reset();
                let mut convergence_samples = 0;

                for i in 0..num_samples {
                    let (filtered_pitch, _) = filter.apply(black_box(step_value), black_box(0.0));

                    // Check if converged (within 5% of target)
                    if (filtered_pitch - step_value).abs() < step_value * 0.05 {
                        convergence_samples = i;
                        break;
                    }
                }

                black_box(convergence_samples)
            });
        });
    }

    group.finish();
}

fn benchmark_outlier_rejection(c: &mut Criterion) {
    let mut group = c.benchmark_group("outlier_rejection");

    // Create signal with periodic outliers
    let mut signal = vec![(0.0, 0.0); 100];
    for i in 0..100 {
        signal[i] = if i % 20 == 10 {
            (100.0, -100.0) // Outlier
        } else {
            (10.0, 10.0) // Normal value
        };
    }

    let filter_configs: Vec<(&str, Box<dyn CursorFilter>)> = vec![
        ("median_5", Box::new(MedianFilter::new(5))),
        ("hampel_7", Box::new(HampelFilter::new(7, 3.0))),
        ("moving_avg_5", Box::new(MovingAverageFilter::new(5))),
        ("kalman", Box::new(KalmanFilter::new())),
    ];

    for (name, mut filter) in filter_configs {
        group.bench_with_input(BenchmarkId::new("process_outliers", name), &signal, |b, signal_data| {
            b.iter(|| {
                filter.reset();
                let mut max_deviation = 0.0f64;

                for &(pitch, yaw) in signal_data {
                    let (fp, fy) = filter.apply(black_box(pitch), black_box(yaw));

                    // Measure how much the filter deviates from the normal value
                    let deviation = ((fp - 10.0).abs() + (fy - 10.0).abs()) / 2.0;
                    max_deviation = max_deviation.max(deviation);
                }

                black_box(max_deviation)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_filter_accuracy,
    benchmark_filter_convergence,
    benchmark_outlier_rejection
);
criterion_main!(benches);
