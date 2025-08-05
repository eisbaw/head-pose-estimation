//! Safe casting utilities to prevent overflow on 32-bit systems

use crate::{Error, Result};

/// Safely convert usize to i32 with overflow checking
pub fn usize_to_i32(value: usize) -> Result<i32> {
    value.try_into()
        .map_err(|_| Error::InvalidInput(format!(
            "Value {} too large to fit in i32", value
        )))
}

/// Safely convert u32 to i32 with overflow checking
pub fn u32_to_i32(value: u32) -> Result<i32> {
    value.try_into()
        .map_err(|_| Error::InvalidInput(format!(
            "Value {} too large to fit in i32", value
        )))
}

/// Safely convert f32 to i32 with bounds checking
pub fn f32_to_i32(value: f32) -> Result<i32> {
    if value.is_finite() && value >= i32::MIN as f32 && value <= i32::MAX as f32 {
        Ok(value as i32)
    } else {
        Err(Error::InvalidInput(format!(
            "Value {} cannot be safely converted to i32", value
        )))
    }
}

/// Safely convert f64 to i32 with bounds checking  
pub fn f64_to_i32(value: f64) -> Result<i32> {
    if value.is_finite() && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
        Ok(value as i32)
    } else {
        Err(Error::InvalidInput(format!(
            "Value {} cannot be safely converted to i32", value
        )))
    }
}

/// Clamp and convert f32 to i32 for pixel coordinates
pub fn f32_to_i32_clamp(value: f32, min: i32, max: i32) -> i32 {
    if !value.is_finite() {
        return min;
    }
    value.clamp(min as f32, max as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_usize_to_i32() {
        assert_eq!(usize_to_i32(42).unwrap(), 42);
        assert_eq!(usize_to_i32(0).unwrap(), 0);
        assert_eq!(usize_to_i32(i32::MAX as usize).unwrap(), i32::MAX);
        
        // On 64-bit systems, this should fail
        if std::mem::size_of::<usize>() > 4 {
            assert!(usize_to_i32(i32::MAX as usize + 1).is_err());
        }
    }
    
    #[test]
    fn test_f32_to_i32() {
        assert_eq!(f32_to_i32(42.0).unwrap(), 42);
        assert_eq!(f32_to_i32(-42.0).unwrap(), -42);
        assert_eq!(f32_to_i32(0.0).unwrap(), 0);
        
        assert!(f32_to_i32(f32::INFINITY).is_err());
        assert!(f32_to_i32(f32::NEG_INFINITY).is_err());
        assert!(f32_to_i32(f32::NAN).is_err());
        assert!(f32_to_i32(i32::MAX as f32 * 2.0).is_err());
    }
    
    #[test]
    fn test_f32_to_i32_clamp() {
        assert_eq!(f32_to_i32_clamp(50.0, 0, 100), 50);
        assert_eq!(f32_to_i32_clamp(-10.0, 0, 100), 0);
        assert_eq!(f32_to_i32_clamp(150.0, 0, 100), 100);
        assert_eq!(f32_to_i32_clamp(f32::NAN, 0, 100), 0);
    }
}