use opencv::{
    core::{Mat, Point2f, Point3f, Vec3d},
    calib3d,
    prelude::*,
};
use std::path::Path;
use std::fs;
use crate::{Result, Error};

/// Head pose estimator using `PnP` algorithm
pub struct PoseEstimator {
    model_points: Vec<Point3f>,
    camera_matrix: Mat,
    dist_coeffs: Mat,
}

impl PoseEstimator {
    /// Create a new pose estimator with 3D model points and camera parameters
    pub fn new<P: AsRef<Path>>(model_path: P, image_width: i32, image_height: i32) -> Result<Self> {
        // Load 3D model points from file
        let model_content = fs::read_to_string(model_path)?;
        let model_points = Self::parse_model_points(&model_content)?;
        
        // Initialize camera matrix with typical values
        let focal_length = f64::from(image_width);
        let center = (f64::from(image_width) / 2.0, f64::from(image_height) / 2.0);
        
        // Create camera matrix using zeros and then fill it
        let mut camera_matrix = Mat::zeros(3, 3, opencv::core::CV_64F)?.to_mat()?;
        let camera_matrix_data: [f64; 9] = [
            focal_length, 0.0, center.0,
            0.0, focal_length, center.1,
            0.0, 0.0, 1.0,
        ];
        
        // Fill camera matrix
        for i in 0..3 {
            for j in 0..3 {
                *camera_matrix.at_2d_mut::<f64>(i, j)? = camera_matrix_data[(i * 3 + j) as usize];
            }
        }
        
        // Assume no lens distortion
        let dist_coeffs = Mat::zeros(4, 1, opencv::core::CV_64F)?.to_mat()?;
        
        Ok(Self {
            model_points,
            camera_matrix,
            dist_coeffs,
        })
    }
    
    /// Estimate head pose from 68 facial landmarks
    pub fn estimate_pose(&self, landmarks: &[(f32, f32)]) -> Result<(Vec3d, Vec3d, Mat)> {
        if landmarks.len() != 68 {
            return Err(Error::InvalidInput(
                format!("Expected 68 landmarks, got {}", landmarks.len())
            ));
        }
        
        // Convert landmarks to OpenCV points
        let image_points: Vec<Point2f> = landmarks.iter()
            .map(|&(x, y)| Point2f::new(x, y))
            .collect();
        
        // Convert points to Mat
        let mut object_points_mat = Mat::zeros(self.model_points.len() as i32, 3, opencv::core::CV_32F)?.to_mat()?;
        for (i, point) in self.model_points.iter().enumerate() {
            *object_points_mat.at_2d_mut::<f32>(i as i32, 0)? = point.x;
            *object_points_mat.at_2d_mut::<f32>(i as i32, 1)? = point.y;
            *object_points_mat.at_2d_mut::<f32>(i as i32, 2)? = point.z;
        }
        
        let mut image_points_mat = Mat::zeros(image_points.len() as i32, 2, opencv::core::CV_32F)?.to_mat()?;
        for (i, point) in image_points.iter().enumerate() {
            *image_points_mat.at_2d_mut::<f32>(i as i32, 0)? = point.x;
            *image_points_mat.at_2d_mut::<f32>(i as i32, 1)? = point.y;
        }
        
        // Solve PnP problem
        let mut rvec = Mat::default();
        let mut tvec = Mat::default();
        
        calib3d::solve_pnp(
            &object_points_mat,
            &image_points_mat,
            &self.camera_matrix,
            &self.dist_coeffs,
            &mut rvec,
            &mut tvec,
            false,
            calib3d::SOLVEPNP_ITERATIVE,
        )?;
        
        // Convert rotation vector to rotation matrix
        let mut rotation_matrix = Mat::default();
        calib3d::rodrigues(&rvec, &mut rotation_matrix, &mut Mat::default())?;
        
        // Extract Euler angles
        let _euler_angles = Self::rotation_matrix_to_euler(&rotation_matrix)?;
        
        Ok((
            Vec3d::from([*rvec.at_2d::<f64>(0, 0)?, *rvec.at_2d::<f64>(1, 0)?, *rvec.at_2d::<f64>(2, 0)?]),
            Vec3d::from([*tvec.at_2d::<f64>(0, 0)?, *tvec.at_2d::<f64>(1, 0)?, *tvec.at_2d::<f64>(2, 0)?]),
            rotation_matrix,
        ))
    }
    
    /// Parse 3D model points from text file
    fn parse_model_points(content: &str) -> Result<Vec<Point3f>> {
        let mut points = Vec::new();
        
        for line in content.lines() {
            let coords: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
                
            if coords.len() == 3 {
                points.push(Point3f::new(coords[0], coords[1], coords[2]));
            }
        }
        
        if points.len() != 68 {
            return Err(Error::ModelError(
                format!("Expected 68 model points, got {}", points.len())
            ));
        }
        
        Ok(points)
    }
    
    /// Convert rotation matrix to Euler angles
    fn rotation_matrix_to_euler(rotation_matrix: &Mat) -> Result<Vec3d> {
        // Extract rotation matrix values
        let _r11 = rotation_matrix.at_2d::<f64>(0, 0)?;
        let _r12 = rotation_matrix.at_2d::<f64>(0, 1)?;
        let r13 = rotation_matrix.at_2d::<f64>(0, 2)?;
        let r21 = rotation_matrix.at_2d::<f64>(1, 0)?;
        let r22 = rotation_matrix.at_2d::<f64>(1, 1)?;
        let r23 = rotation_matrix.at_2d::<f64>(1, 2)?;
        let _r31 = rotation_matrix.at_2d::<f64>(2, 0)?;
        let _r32 = rotation_matrix.at_2d::<f64>(2, 1)?;
        let r33 = rotation_matrix.at_2d::<f64>(2, 2)?;
        
        // Calculate Euler angles (in radians)
        let pitch = (-r23).asin();
        let yaw = r13.atan2(*r33);
        let roll = r21.atan2(*r22);
        
        // Convert to degrees
        Ok(Vec3d::from([
            pitch.to_degrees(),
            yaw.to_degrees(),
            roll.to_degrees(),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_angle_conversion() {
        // Test with identity matrix
        let identity = Mat::eye(3, 3, opencv::core::CV_64F).unwrap().to_mat().unwrap();
        let angles = PoseEstimator::rotation_matrix_to_euler(&identity).unwrap();
        
        assert!((angles[0]).abs() < 1e-6);
        assert!((angles[1]).abs() < 1e-6);
        assert!((angles[2]).abs() < 1e-6);
    }
}