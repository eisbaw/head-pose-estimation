//! Benchmarks for pose estimation and utility functions

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use head_pose_estimation::{
    pose_estimation::PoseEstimator,
    utils::{refine_boxes, safe_cast::*},
};
use opencv::core::Rect;

fn benchmark_pose_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pose_estimation");

    // Create pose estimator with test model
    let pose_estimator = PoseEstimator::new("assets/model.txt", 640, 480).expect("Failed to create pose estimator");

    // Generate test landmarks (68 points)
    let test_landmarks: Vec<(f32, f32)> = (0..68)
        .map(|i| {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / 68.0;
            let x = 320.0 + 100.0 * angle.cos();
            let y = 240.0 + 100.0 * angle.sin();
            (x, y)
        })
        .collect();

    group.bench_function("estimate_pose_68_landmarks", |b| {
        b.iter(|| {
            let result = pose_estimator
                .estimate_pose(&test_landmarks)
                .expect("Pose estimation failed");
            black_box(result);
        });
    });

    // Benchmark Euler angle conversion
    group.bench_function("euler_angle_conversion", |b| {
        use opencv::core::Mat;

        // Create OpenCV Mat from rotation values
        let rotation_data: Vec<f64> = vec![0.9, -0.1, 0.4, 0.1, 0.99, 0.05, -0.4, 0.0, 0.9];

        let rotation_matrix =
            Mat::from_slice_2d(&[&rotation_data[0..3], &rotation_data[3..6], &rotation_data[6..9]]).unwrap();

        b.iter(|| {
            let angles =
                PoseEstimator::rotation_matrix_to_euler(&rotation_matrix).expect("Failed to convert rotation matrix");
            black_box(angles);
        });
    });

    group.finish();
}

fn benchmark_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    // Benchmark refine_boxes
    let test_boxes = vec![
        Rect::new(10, 20, 100, 150),
        Rect::new(50, 60, 200, 250),
        Rect::new(100, 120, 80, 90),
        Rect::new(200, 180, 150, 160),
    ];

    group.bench_function("refine_boxes_4", |b| {
        b.iter(|| {
            let mut boxes = test_boxes.clone();
            refine_boxes(&mut boxes, 640, 480, 0.25).expect("Failed to refine boxes");
            black_box(boxes);
        });
    });

    // Benchmark safe cast functions
    group.bench_function("safe_cast_f32_to_i32", |b| {
        let values: Vec<f32> = vec![10.5, -20.3, 100000.0, -100000.0, std::f32::NAN];
        b.iter(|| {
            for &val in &values {
                let result = f32_to_i32(val).unwrap_or(0);
                black_box(result);
            }
        });
    });

    group.bench_function("safe_cast_f32_to_i32_clamp", |b| {
        let values: Vec<f32> = vec![10.5, -20.3, 1e10, -1e10, std::f32::NAN];
        b.iter(|| {
            for &val in &values {
                black_box(f32_to_i32_clamp(val, i32::MIN, i32::MAX));
            }
        });
    });

    group.bench_function("safe_cast_usize_to_i32", |b| {
        let values: Vec<usize> = vec![0, 100, 1000, 10000, usize::MAX];
        b.iter(|| {
            for &val in &values {
                black_box(usize_to_i32(val).unwrap_or(i32::MAX));
            }
        });
    });

    group.finish();
}

fn benchmark_movement_detection(c: &mut Criterion) {
    use head_pose_estimation::movement_detector::MovementDetector;

    let mut group = c.benchmark_group("movement_detection");

    let mut detector = MovementDetector::new(30, 0.5);

    // Generate test movement data
    let stationary_data: Vec<(f64, f64)> = (0..30)
        .map(|_| (10.0 + 0.1 * rand::random::<f64>(), 15.0 + 0.1 * rand::random::<f64>()))
        .collect();

    let moving_data: Vec<(f64, f64)> = (0..30)
        .map(|i| {
            let t = i as f64 * 0.1;
            (10.0 + 5.0 * t.sin(), 15.0 + 5.0 * t.cos())
        })
        .collect();

    group.bench_function("detect_stationary", |b| {
        b.iter(|| {
            detector.reset();
            let mut result = false;
            for &(pitch, yaw) in &stationary_data {
                result = detector.update(pitch, yaw);
            }
            black_box(result);
        });
    });

    group.bench_function("detect_movement", |b| {
        b.iter(|| {
            detector.reset();
            let mut result = false;
            for &(pitch, yaw) in &moving_data {
                result = detector.update(pitch, yaw);
            }
            black_box(result);
        });
    });

    group.bench_function("get_stats", |b| {
        // Pre-fill detector
        for &(pitch, yaw) in &stationary_data {
            detector.update(pitch, yaw);
        }

        b.iter(|| {
            let stats = detector.get_stats();
            black_box(stats);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_pose_estimation,
    benchmark_utils,
    benchmark_movement_detection
);
criterion_main!(benches);
