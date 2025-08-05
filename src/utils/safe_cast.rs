//! Safe casting utilities to prevent overflow on 32-bit systems

use crate::{Error, Result};

/// Safely convert usize to i32 with overflow checking
///
/// # Errors
///
/// Returns an error if the value exceeds i32::MAX
pub fn usize_to_i32(value: usize) -> Result<i32> {
    value
        .try_into()
        .map_err(|_| Error::InvalidInput(format!("Value {value} too large to fit in i32")))
}

/// Safely convert u32 to i32 with overflow checking
///
/// # Errors
///
/// Returns an error if the value exceeds i32::MAX
pub fn u32_to_i32(value: u32) -> Result<i32> {
    value
        .try_into()
        .map_err(|_| Error::InvalidInput(format!("Value {value} too large to fit in i32")))
}

/// Safely convert f32 to i32 with bounds checking
///
/// # Errors
///
/// Returns an error if the value is not finite or outside i32 range
#[allow(clippy::cast_precision_loss)] // MIN/MAX bounds checking is approximate
#[allow(clippy::cast_possible_truncation)] // Truncation after bounds check is safe
pub fn f32_to_i32(value: f32) -> Result<i32> {
    if value.is_finite() && value >= i32::MIN as f32 && value <= i32::MAX as f32 {
        Ok(value as i32)
    } else {
        Err(Error::InvalidInput(format!(
            "Value {value} cannot be safely converted to i32"
        )))
    }
}

/// Safely convert f64 to i32 with bounds checking
///
/// # Errors
///
/// Returns an error if the value is not finite or outside i32 range
#[allow(clippy::cast_possible_truncation)] // Truncation after bounds check is safe
pub fn f64_to_i32(value: f64) -> Result<i32> {
    if value.is_finite() && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX) {
        Ok(value as i32)
    } else {
        Err(Error::InvalidInput(format!(
            "Value {value} cannot be safely converted to i32"
        )))
    }
}

/// Clamp and convert f32 to i32 for pixel coordinates
#[must_use]
#[allow(clippy::cast_precision_loss)] // Acceptable for clamping bounds
#[allow(clippy::cast_possible_truncation)] // Clamping ensures safe truncation
pub fn f32_to_i32_clamp(value: f32, min: i32, max: i32) -> i32 {
    // Ensure min <= max
    let (min, max) = if min <= max { (min, max) } else { (max, min) };

    if !value.is_finite() {
        return min;
    }

    // Convert bounds to f32 and clamp
    let clamped = value.clamp(min as f32, max as f32);

    // Ensure result is within bounds after conversion
    let result = clamped as i32;
    result.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    #[test]
    fn test_u32_to_i32() {
        assert_eq!(u32_to_i32(42).unwrap(), 42);
        assert_eq!(u32_to_i32(0).unwrap(), 0);
        assert_eq!(u32_to_i32(i32::MAX as u32).unwrap(), i32::MAX);
        assert!(u32_to_i32(i32::MAX as u32 + 1).is_err());
        assert!(u32_to_i32(u32::MAX).is_err());
    }

    #[test]
    fn test_f64_to_i32() {
        assert_eq!(f64_to_i32(42.0).unwrap(), 42);
        assert_eq!(f64_to_i32(-42.0).unwrap(), -42);
        assert_eq!(f64_to_i32(0.0).unwrap(), 0);
        assert_eq!(f64_to_i32(2147483647.0).unwrap(), i32::MAX);
        assert_eq!(f64_to_i32(-2147483648.0).unwrap(), i32::MIN);

        assert!(f64_to_i32(f64::INFINITY).is_err());
        assert!(f64_to_i32(f64::NEG_INFINITY).is_err());
        assert!(f64_to_i32(f64::NAN).is_err());
        assert!(f64_to_i32(2147483648.0).is_err());
        assert!(f64_to_i32(-2147483649.0).is_err());
    }

    #[test]
    fn test_edge_case_rounding() {
        // Test values very close to boundaries
        assert_eq!(f32_to_i32(2147483520.0).unwrap(), 2147483520);
        assert_eq!(f64_to_i32(2147483646.99).unwrap(), 2147483646);

        // Test small fractions
        assert_eq!(f32_to_i32(0.9).unwrap(), 0);
        assert_eq!(f32_to_i32(-0.9).unwrap(), 0);
        assert_eq!(f64_to_i32(0.9999).unwrap(), 0);
        assert_eq!(f64_to_i32(-0.9999).unwrap(), 0);
    }

    #[test]
    fn test_f32_to_i32_clamp_edge_cases() {
        // Test with extreme bounds
        assert_eq!(f32_to_i32_clamp(50.0, i32::MIN, i32::MAX), 50);
        assert_eq!(f32_to_i32_clamp(f32::INFINITY, 0, 100), 0); // Non-finite returns min
        assert_eq!(f32_to_i32_clamp(f32::NEG_INFINITY, 0, 100), 0); // Non-finite returns min

        // Test with negative bounds
        assert_eq!(f32_to_i32_clamp(-50.0, -100, -10), -50);
        assert_eq!(f32_to_i32_clamp(-150.0, -100, -10), -100);
        assert_eq!(f32_to_i32_clamp(0.0, -100, -10), -10);

        // Test with identical bounds
        assert_eq!(f32_to_i32_clamp(50.0, 42, 42), 42);
        assert_eq!(f32_to_i32_clamp(f32::NAN, 42, 42), 42);
    }

    // Property-based tests
    proptest! {
        #[test]
        fn prop_usize_to_i32_within_bounds(value in 0..=i32::MAX as usize) {
            let result = usize_to_i32(value);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap() as usize, value);
        }

        #[test]
        fn prop_u32_to_i32_within_bounds(value in 0..=i32::MAX as u32) {
            let result = u32_to_i32(value);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap() as u32, value);
        }

        #[test]
        fn prop_f32_to_i32_finite_within_bounds(value in any::<i32>()) {
            let f_value = value as f32;
            if f_value as i32 == value {  // Check if conversion is lossless
                let result = f32_to_i32(f_value);
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), value);
            }
        }

        #[test]
        fn prop_f64_to_i32_finite_within_bounds(value in i32::MIN..=i32::MAX) {
            let f_value = f64::from(value);
            let result = f64_to_i32(f_value);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap(), value);
        }

        #[test]
        fn prop_f32_to_i32_clamp_always_within_bounds(
            value in any::<f32>(),
            min in any::<i32>(),
            max in any::<i32>()
        ) {
            let (min, max) = if min <= max { (min, max) } else { (max, min) };
            let result = f32_to_i32_clamp(value, min, max);
            prop_assert!(result >= min);
            prop_assert!(result <= max);
        }

        #[test]
        fn prop_f32_to_i32_clamp_preserves_finite_values(
            min in i32::MIN/2..=0i32,
            max in 0..=i32::MAX/2
        ) {
            let value = (min + max) as f32 / 2.0;
            let result = f32_to_i32_clamp(value, min, max);
            prop_assert!(result >= min);
            prop_assert!(result <= max);
            prop_assert!((result as f32 - value).abs() < 1.0);
        }

        #[test]
        fn prop_cast_consistency(value in 0..1000000i32) {
            // Test consistency across different cast paths
            let as_usize = value as usize;
            let as_u32 = value as u32;
            let as_f32 = value as f32;
            let as_f64 = f64::from(value);

            prop_assert_eq!(usize_to_i32(as_usize).unwrap(), value);
            prop_assert_eq!(u32_to_i32(as_u32).unwrap(), value);
            if as_f32 as i32 == value {  // Only if f32 conversion is lossless
                prop_assert_eq!(f32_to_i32(as_f32).unwrap(), value);
            }
            prop_assert_eq!(f64_to_i32(as_f64).unwrap(), value);
        }
    }
}
