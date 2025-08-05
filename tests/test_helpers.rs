//! Helper functions and utilities for tests

use head_pose_estimation::Result;
use opencv::{core::Mat, prelude::*};

/// Create a test image with specified dimensions and type
pub fn create_test_image(height: i32, width: i32, cv_type: i32) -> Result<Mat> {
    Mat::zeros(height, width, cv_type)?.to_mat().map_err(Into::into)
}

/// Assert that a Vec3d contains finite values
pub fn assert_vec3d_finite(vec: &opencv::core::Vec3d) -> Result<()> {
    for i in 0..3 {
        let value = vec
            .get(i)
            .ok_or_else(|| head_pose_estimation::Error::InvalidInput(format!("Cannot access element {i} of Vec3d")))?;

        if !value.is_finite() {
            return Err(head_pose_estimation::Error::InvalidInput(format!(
                "Non-finite value at index {i}: {value}"
            )));
        }
    }
    Ok(())
}

/// Create a filter safely with error propagation
pub fn create_test_filter(name: &str) -> Result<Box<dyn head_pose_estimation::filters::CursorFilter>> {
    head_pose_estimation::filters::create_filter(name)
}
