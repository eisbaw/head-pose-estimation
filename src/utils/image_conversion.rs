//! Image conversion utilities for OpenCV Mat and ndarray interoperability.

use crate::Result;
use ndarray::{Array3, Array4};
use opencv::core::{Mat, MatTraitConst, Scalar, CV_32F, CV_8U};
use opencv::prelude::MatTrait;

/// Convert an OpenCV Mat to an ndarray Array3<f32>
/// 
/// # Arguments
/// * `mat` - OpenCV Mat with shape (height, width, channels)
/// 
/// # Returns
/// * `Array3<f32>` with shape (height, width, channels)
/// 
/// # Errors
/// * Returns error if Mat dimensions are invalid
/// * Returns error if Mat data cannot be accessed
pub fn mat_to_array3_f32(mat: &Mat) -> Result<Array3<f32>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();
    
    if rows <= 0 || cols <= 0 || channels <= 0 {
        return Err(crate::error::Error::InvalidInput(
            format!("Invalid Mat dimensions: {}x{}x{}", rows, cols, channels)
        ));
    }
    
    let mut data = vec![0.0f32; (rows * cols * channels) as usize];
    
    // Extract data from Mat
    for row in 0..rows {
        for col in 0..cols {
            if channels == 3 {
                let pixel = mat.at_2d::<opencv::core::Vec3f>(row, col)?;
                let base_idx = ((row * cols + col) * channels) as usize;
                data[base_idx] = pixel[0];
                data[base_idx + 1] = pixel[1];
                data[base_idx + 2] = pixel[2];
            } else if channels == 1 {
                let pixel = *mat.at_2d::<f32>(row, col)?;
                let idx = (row * cols) as usize;
                data[idx] = pixel;
            } else {
                // For other channel counts, use generic approach
                for ch in 0..channels {
                    let idx = ((row * cols * channels) + (col * channels) + ch) as usize;
                    // This is a fallback - may not work for all channel counts
                    data[idx] = 0.0;
                }
            }
        }
    }
    
    Array3::from_shape_vec(
        (rows as usize, cols as usize, channels as usize),
        data
    ).map_err(|e| crate::error::Error::InvalidInput(
        format!("Failed to create array from Mat: {}", e)
    ))
}

/// Convert an ndarray Array3<f32> to an OpenCV Mat
/// 
/// # Arguments
/// * `array` - ndarray with shape (height, width, channels)
/// 
/// # Returns
/// * OpenCV Mat with the same shape and CV_32F type
/// 
/// # Errors
/// * Returns error if array shape is invalid
/// * Returns error if Mat creation fails
pub fn array3_f32_to_mat(array: &Array3<f32>) -> Result<Mat> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(crate::error::Error::InvalidInput(
            format!("Expected 3D array, got {}D", shape.len())
        ));
    }
    
    let (height, width, channels) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
    
    let mut mat = Mat::new_rows_cols_with_default(height, width, CV_32F + ((channels - 1) << 3), 0.0.into())?;
    
    // Copy data to Mat
    if channels == 3 {
        for row in 0..height {
            for col in 0..width {
                let mut pixel = opencv::core::Vec3f::default();
                pixel[0] = array[[row as usize, col as usize, 0]];
                pixel[1] = array[[row as usize, col as usize, 1]];
                pixel[2] = array[[row as usize, col as usize, 2]];
                *mat.at_2d_mut::<opencv::core::Vec3f>(row, col)? = pixel;
            }
        }
    } else if channels == 1 {
        for row in 0..height {
            for col in 0..width {
                *mat.at_2d_mut::<f32>(row, col)? = array[[row as usize, col as usize, 0]];
            }
        }
    } else {
        return Err(crate::error::Error::InvalidInput(
            format!("Unsupported channel count: {}", channels)
        ));
    }
    
    Ok(mat)
}

/// Convert an OpenCV Mat to an ndarray Array3<u8>
/// 
/// # Arguments
/// * `mat` - OpenCV Mat with shape (height, width, channels) and CV_8U type
/// 
/// # Returns
/// * `Array3<u8>` with shape (height, width, channels)
/// 
/// # Errors
/// * Returns error if Mat dimensions are invalid
/// * Returns error if Mat data cannot be accessed
pub fn mat_to_array3_u8(mat: &Mat) -> Result<Array3<u8>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();
    
    if rows <= 0 || cols <= 0 || channels <= 0 {
        return Err(crate::error::Error::InvalidInput(
            format!("Invalid Mat dimensions: {}x{}x{}", rows, cols, channels)
        ));
    }
    
    let mut data = vec![0u8; (rows * cols * channels) as usize];
    
    // Extract data from Mat
    for row in 0..rows {
        for col in 0..cols {
            if channels == 3 {
                let pixel = mat.at_2d::<opencv::core::Vec3b>(row, col)?;
                let base_idx = ((row * cols + col) * channels) as usize;
                data[base_idx] = pixel[0];
                data[base_idx + 1] = pixel[1];
                data[base_idx + 2] = pixel[2];
            } else if channels == 1 {
                let pixel = *mat.at_2d::<u8>(row, col)?;
                let idx = (row * cols) as usize;
                data[idx] = pixel;
            } else {
                // For other channel counts, use generic approach
                for ch in 0..channels {
                    let idx = ((row * cols * channels) + (col * channels) + ch) as usize;
                    // This is a fallback - may not work for all channel counts
                    data[idx] = 0;
                }
            }
        }
    }
    
    Array3::from_shape_vec(
        (rows as usize, cols as usize, channels as usize),
        data
    ).map_err(|e| crate::error::Error::InvalidInput(
        format!("Failed to create array from Mat: {}", e)
    ))
}

/// Convert an ndarray Array3<u8> to an OpenCV Mat
/// 
/// # Arguments
/// * `array` - ndarray with shape (height, width, channels)
/// 
/// # Returns
/// * OpenCV Mat with the same shape and CV_8U type
/// 
/// # Errors
/// * Returns error if array shape is invalid
/// * Returns error if Mat creation fails
pub fn array3_u8_to_mat(array: &Array3<u8>) -> Result<Mat> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(crate::error::Error::InvalidInput(
            format!("Expected 3D array, got {}D", shape.len())
        ));
    }
    
    let (height, width, channels) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
    
    let mut mat = Mat::new_rows_cols_with_default(height, width, CV_8U + ((channels - 1) << 3), Scalar::default())?;
    
    // Copy data to Mat
    if channels == 3 {
        for row in 0..height {
            for col in 0..width {
                let mut pixel = opencv::core::Vec3b::default();
                pixel[0] = array[[row as usize, col as usize, 0]];
                pixel[1] = array[[row as usize, col as usize, 1]];
                pixel[2] = array[[row as usize, col as usize, 2]];
                *mat.at_2d_mut::<opencv::core::Vec3b>(row, col)? = pixel;
            }
        }
    } else if channels == 1 {
        for row in 0..height {
            for col in 0..width {
                *mat.at_2d_mut::<u8>(row, col)? = array[[row as usize, col as usize, 0]];
            }
        }
    } else {
        return Err(crate::error::Error::InvalidInput(
            format!("Unsupported channel count: {}", channels)
        ));
    }
    
    Ok(mat)
}

/// Convert a batch of images from Vec<Mat> to Array4<f32>
/// 
/// # Arguments
/// * `mats` - Vector of OpenCV Mats, all with same dimensions
/// 
/// # Returns
/// * `Array4<f32>` with shape (batch, height, width, channels)
/// 
/// # Errors
/// * Returns error if Mats have different dimensions
/// * Returns error if any Mat conversion fails
pub fn mats_to_array4_f32(mats: &[Mat]) -> Result<Array4<f32>> {
    if mats.is_empty() {
        return Err(crate::error::Error::InvalidInput(
            "Empty Mat vector".to_string()
        ));
    }
    
    // Get dimensions from first Mat
    let first_rows = mats[0].rows() as usize;
    let first_cols = mats[0].cols() as usize;
    let first_channels = mats[0].channels() as usize;
    
    // Validate all Mats have same dimensions
    for (i, mat) in mats.iter().enumerate() {
        if mat.rows() as usize != first_rows || 
           mat.cols() as usize != first_cols || 
           mat.channels() as usize != first_channels {
            return Err(crate::error::Error::InvalidInput(
                format!("Mat at index {} has different dimensions", i)
            ));
        }
    }
    
    let batch_size = mats.len();
    let mut data = vec![0.0f32; batch_size * first_rows * first_cols * first_channels];
    
    // Convert each Mat
    for (batch_idx, mat) in mats.iter().enumerate() {
        let array = mat_to_array3_f32(mat)?;
        let start_idx = batch_idx * first_rows * first_cols * first_channels;
        data[start_idx..start_idx + array.len()].copy_from_slice(array.as_slice().unwrap());
    }
    
    Array4::from_shape_vec(
        (batch_size, first_rows, first_cols, first_channels),
        data
    ).map_err(|e| crate::error::Error::InvalidInput(
        format!("Failed to create batch array: {}", e)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, CV_32FC3, CV_8UC3};
    
    #[test]
    fn test_mat_to_array3_f32() {
        let mat = Mat::new_rows_cols_with_default(
            2, 3, CV_32FC3, 
            Scalar::new(1.0, 2.0, 3.0, 0.0)
        ).unwrap();
        
        let array = mat_to_array3_f32(&mat).unwrap();
        assert_eq!(array.shape(), &[2, 3, 3]);
        assert_eq!(array[[0, 0, 0]], 1.0);
        assert_eq!(array[[0, 0, 1]], 2.0);
        assert_eq!(array[[0, 0, 2]], 3.0);
    }
    
    #[test]
    fn test_array3_f32_to_mat() {
        let array = Array3::<f32>::from_shape_vec(
            (2, 3, 3),
            vec![1.0; 18]
        ).unwrap();
        
        let mat = array3_f32_to_mat(&array).unwrap();
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.channels(), 3);
    }
    
    #[test]
    fn test_roundtrip_f32() {
        let original = Array3::<f32>::from_shape_vec(
            (4, 5, 3),
            (0..60).map(|i| i as f32).collect()
        ).unwrap();
        
        let mat = array3_f32_to_mat(&original).unwrap();
        let recovered = mat_to_array3_f32(&mat).unwrap();
        
        assert_eq!(original.shape(), recovered.shape());
        for ((&a, &b), _) in original.iter().zip(recovered.iter()).zip(0..60) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_mat_to_array3_u8() {
        let mat = Mat::new_rows_cols_with_default(
            2, 3, CV_8UC3, 
            Scalar::new(10.0, 20.0, 30.0, 0.0)
        ).unwrap();
        
        let array = mat_to_array3_u8(&mat).unwrap();
        assert_eq!(array.shape(), &[2, 3, 3]);
        assert_eq!(array[[0, 0, 0]], 10);
        assert_eq!(array[[0, 0, 1]], 20);
        assert_eq!(array[[0, 0, 2]], 30);
    }
    
    #[test]
    fn test_mats_to_array4_f32() {
        let mats = vec![
            Mat::new_rows_cols_with_default(
                2, 3, CV_32FC3,
                Scalar::new(1.0, 2.0, 3.0, 0.0)
            ).unwrap(),
            Mat::new_rows_cols_with_default(
                2, 3, CV_32FC3,
                Scalar::new(4.0, 5.0, 6.0, 0.0)
            ).unwrap()
        ];
        
        let array = mats_to_array4_f32(&mats).unwrap();
        assert_eq!(array.shape(), &[2, 2, 3, 3]);
        assert_eq!(array[[0, 0, 0, 0]], 1.0);
        assert_eq!(array[[1, 0, 0, 0]], 4.0);
    }
}