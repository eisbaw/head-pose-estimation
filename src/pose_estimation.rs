use crate::{
    constants::{CAMERA_CENTER_FACTOR, MODEL_POINTS_TOTAL_VALUES, NUM_FACIAL_LANDMARKS},
    utils::safe_cast::usize_to_i32,
    Error, Result,
};
use opencv::{
    calib3d,
    core::{Mat, Point2f, Point2i, Point3f, Scalar, Vec3d},
    prelude::*,
};
use std::fs;
use std::path::Path;

/// Head pose estimator using `PnP` algorithm
pub struct PoseEstimator {
    model_points: Vec<Point3f>,
    camera_matrix: Mat,
    dist_coeffs: Mat,
}

impl PoseEstimator {
    /// Create a new pose estimator with 3D model points and camera parameters
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model file cannot be read
    /// - The model file has an invalid format
    /// - OpenCV matrix operations fail
    pub fn new<P: AsRef<Path>>(model_path: P, image_width: i32, image_height: i32) -> Result<Self> {
        log::info!(
            "Initializing PoseEstimator with model: {}",
            model_path.as_ref().display()
        );
        // Load 3D model points from file
        let model_content = fs::read_to_string(model_path)?;
        let model_points = Self::parse_model_points(&model_content)?;

        // Initialize camera matrix with typical values
        let focal_length = f64::from(image_width);
        let center = (
            f64::from(image_width) / CAMERA_CENTER_FACTOR,
            f64::from(image_height) / CAMERA_CENTER_FACTOR,
        );

        // Create camera matrix using zeros and then fill it
        let mut camera_matrix = Mat::zeros(3, 3, opencv::core::CV_64F)?.to_mat()?;
        let camera_matrix_data: [f64; 9] = [focal_length, 0.0, center.0, 0.0, focal_length, center.1, 0.0, 0.0, 1.0];

        // Fill camera matrix
        for (idx, &value) in camera_matrix_data.iter().enumerate() {
            let i = idx / 3;
            let j = idx % 3;
            *camera_matrix.at_2d_mut::<f64>(usize_to_i32(i)?, usize_to_i32(j)?)? = value;
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
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of landmarks is not exactly 68
    /// - The PnP solver fails to converge
    /// - OpenCV operations fail
    pub fn estimate_pose(&self, landmarks: &[(f32, f32)]) -> Result<(Vec3d, Vec3d, Mat)> {
        if landmarks.len() != NUM_FACIAL_LANDMARKS {
            return Err(Error::InvalidInput(format!(
                "Expected {} landmarks, got {}",
                NUM_FACIAL_LANDMARKS,
                landmarks.len()
            )));
        }

        // Convert landmarks to OpenCV points
        let image_points: Vec<Point2f> = landmarks.iter().map(|&(x, y)| Point2f::new(x, y)).collect();

        // Convert points to Mat
        let rows: i32 = self
            .model_points
            .len()
            .try_into()
            .map_err(|_| Error::InvalidInput("Too many model points".to_string()))?;
        let mut object_points_mat = Mat::zeros(rows, 3, opencv::core::CV_32F)?.to_mat()?;
        for (i, point) in self.model_points.iter().enumerate() {
            let idx: i32 = i
                .try_into()
                .map_err(|_| Error::InvalidInput("Index overflow".to_string()))?;
            *object_points_mat.at_2d_mut::<f32>(idx, 0)? = point.x;
            *object_points_mat.at_2d_mut::<f32>(idx, 1)? = point.y;
            *object_points_mat.at_2d_mut::<f32>(idx, 2)? = point.z;
        }

        let rows: i32 = image_points
            .len()
            .try_into()
            .map_err(|_| Error::InvalidInput("Too many image points".to_string()))?;
        let mut image_points_mat = Mat::zeros(rows, 2, opencv::core::CV_32F)?.to_mat()?;
        for (i, point) in image_points.iter().enumerate() {
            let idx: i32 = i
                .try_into()
                .map_err(|_| Error::InvalidInput("Index overflow".to_string()))?;
            *image_points_mat.at_2d_mut::<f32>(idx, 0)? = point.x;
            *image_points_mat.at_2d_mut::<f32>(idx, 1)? = point.y;
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
            Vec3d::from([
                *rvec.at_2d::<f64>(0, 0)?,
                *rvec.at_2d::<f64>(1, 0)?,
                *rvec.at_2d::<f64>(2, 0)?,
            ]),
            Vec3d::from([
                *tvec.at_2d::<f64>(0, 0)?,
                *tvec.at_2d::<f64>(1, 0)?,
                *tvec.at_2d::<f64>(2, 0)?,
            ]),
            rotation_matrix,
        ))
    }

    /// Parse 3D model points from text file
    fn parse_model_points(content: &str) -> Result<Vec<Point3f>> {
        let values: Vec<f32> = content
            .lines()
            .filter_map(|line| line.trim().parse::<f32>().ok())
            .collect();

        if values.len() != MODEL_POINTS_TOTAL_VALUES {
            return Err(Error::ModelValidationError(format!(
                "Expected {} coordinate values ({} points × 3), got {}",
                MODEL_POINTS_TOTAL_VALUES,
                NUM_FACIAL_LANDMARKS,
                values.len()
            )));
        }

        let mut points = Vec::new();
        for i in (0..values.len()).step_by(3) {
            points.push(Point3f::new(values[i], values[i + 1], values[i + 2]));
        }

        Ok(points)
    }

    /// Convert rotation matrix to Euler angles
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix access operations fail
    pub fn rotation_matrix_to_euler(rotation_matrix: &Mat) -> Result<Vec3d> {
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
        Ok(Vec3d::from([pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()]))
    }
    
    /// Visualize pose by drawing a 3D box
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - OpenCV drawing operations fail
    /// - Point projection fails
    pub fn visualize(
        &self,
        image: &mut Mat,
        rotation_vec: &Vec3d,
        translation_vec: &Vec3d,
        color: Scalar,
        line_width: i32,
    ) -> Result<()> {
        // Define 3D box points
        let mut point_3d = Vec::new();
        
        // Rear face (smaller, at origin)
        let rear_size = 75.0;
        let rear_depth = 0.0;
        point_3d.push(Point3f::new(-rear_size, -rear_size, rear_depth));
        point_3d.push(Point3f::new(-rear_size, rear_size, rear_depth));
        point_3d.push(Point3f::new(rear_size, rear_size, rear_depth));
        point_3d.push(Point3f::new(rear_size, -rear_size, rear_depth));
        point_3d.push(Point3f::new(-rear_size, -rear_size, rear_depth)); // Close the rear face
        
        // Front face (larger, forward)
        let front_size = 100.0;
        let front_depth = 100.0;
        point_3d.push(Point3f::new(-front_size, -front_size, front_depth));
        point_3d.push(Point3f::new(-front_size, front_size, front_depth));
        point_3d.push(Point3f::new(front_size, front_size, front_depth));
        point_3d.push(Point3f::new(front_size, -front_size, front_depth));
        point_3d.push(Point3f::new(-front_size, -front_size, front_depth)); // Close the front face
        
        // Convert to Mat for projectPoints
        let object_points = Mat::from_slice(&point_3d)?;
        
        // Project 3D points to 2D
        let mut image_points = Mat::default();
        opencv::calib3d::project_points(
            &object_points,
            rotation_vec,
            translation_vec,
            &self.camera_matrix,
            &self.dist_coeffs,
            &mut image_points,
            &mut Mat::default(),
            0.0,
        )?;
        
        // Convert projected points to Point2i for drawing
        let mut points_2d = Vec::new();
        for i in 0..10 {
            let pt = image_points.at_2d::<opencv::core::Point2d>(i, 0)?;
            points_2d.push(Point2i::new(pt.x as i32, pt.y as i32));
        }
        
        // Draw the rear face using lines
        for i in 0..4 {
            let j = (i + 1) % 4;
            opencv::imgproc::line(
                image,
                points_2d[i],
                points_2d[j],
                color,
                line_width,
                opencv::imgproc::LINE_AA,
                0,
            )?;
        }
        
        // Draw the front face using lines
        for i in 0..4 {
            let j = (i + 1) % 4;
            opencv::imgproc::line(
                image,
                points_2d[i + 5],
                points_2d[j + 5],
                color,
                line_width,
                opencv::imgproc::LINE_AA,
                0,
            )?;
        }
        
        // Draw connecting lines between rear and front faces
        opencv::imgproc::line(
            image,
            points_2d[1],
            points_2d[6],
            color,
            line_width,
            opencv::imgproc::LINE_AA,
            0,
        )?;
        opencv::imgproc::line(
            image,
            points_2d[2],
            points_2d[7],
            color,
            line_width,
            opencv::imgproc::LINE_AA,
            0,
        )?;
        opencv::imgproc::line(
            image,
            points_2d[3],
            points_2d[8],
            color,
            line_width,
            opencv::imgproc::LINE_AA,
            0,
        )?;
        
        Ok(())
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

    #[test]
    fn test_parse_model_points() {
        // Valid model data - must be exactly MODEL_POINTS_TOTAL_VALUES values (NUM_FACIAL_LANDMARKS points × 3)
        let mut values = Vec::new();
        for i in 0..NUM_FACIAL_LANDMARKS {
            values.push(format!("{}.0", i * 3));
            values.push(format!("{}.0", i * 3 + 1));
            values.push(format!("{}.0", i * 3 + 2));
        }
        let valid_data = values.join("\n");
        let points = PoseEstimator::parse_model_points(&valid_data).unwrap();
        assert_eq!(points.len(), NUM_FACIAL_LANDMARKS);
        assert_eq!(points[0].x, 0.0);
        assert_eq!(points[0].y, 1.0);
        assert_eq!(points[0].z, 2.0);

        // Wrong number of values
        let invalid_data = "1.0\n2.0\n3.0";
        assert!(PoseEstimator::parse_model_points(invalid_data).is_err());

        // Empty data
        let empty_data = "";
        assert!(PoseEstimator::parse_model_points(empty_data).is_err());
    }

    #[test]
    fn test_parse_model_points_invalid() {
        // Invalid number format - but the current implementation filters invalid lines
        let mut values = Vec::new();
        for i in 0..67 {
            values.push(format!("{}.0", i));
        }
        values.push("abc".to_string()); // This will be filtered out
        values.push("203.0".to_string());
        let invalid_data = values.join("\n");
        // This will have 203 valid values, not 204
        assert!(PoseEstimator::parse_model_points(&invalid_data).is_err());

        // Too few values
        let wrong_count = (0..200).map(|i| format!("{}.0", i)).collect::<Vec<_>>().join("\n");
        assert!(PoseEstimator::parse_model_points(&wrong_count).is_err());

        // Too many values
        let too_many = (0..205).map(|i| format!("{}.0", i)).collect::<Vec<_>>().join("\n");
        assert!(PoseEstimator::parse_model_points(&too_many).is_err());
    }
}
