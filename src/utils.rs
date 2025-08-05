//! Utility functions for image processing and coordinate transformations.

pub mod safe_cast;
pub mod image_conversion;

use crate::Result;
use opencv::core::Rect;
use safe_cast::f32_to_i32_clamp;

/// Refine bounding boxes to ensure they are within image boundaries
///
/// This is a port of the Python `refine()` function from utils.py
///
/// # Errors
///
/// Currently returns Ok(()) always, but returns Result for API consistency
#[allow(clippy::cast_precision_loss)] // Precision loss acceptable for box dimensions
pub fn refine_boxes(boxes: &mut [Rect], max_width: i32, max_height: i32, shift: f32) -> Result<()> {
    for bbox in boxes.iter_mut() {
        let x_shift = f32_to_i32_clamp(bbox.width as f32 * shift, 0, max_width);
        let y_shift = f32_to_i32_clamp(bbox.height as f32 * shift, 0, max_height);

        // Expand the bounding box
        bbox.x = (bbox.x - x_shift).max(0);
        bbox.y = (bbox.y - y_shift).max(0);
        bbox.width = (bbox.width + 2 * x_shift).min(max_width - bbox.x);
        bbox.height = (bbox.height + 2 * y_shift).min(max_height - bbox.y);

        // Make it square
        let side_length = bbox.width.max(bbox.height);
        bbox.width = side_length;
        bbox.height = side_length;

        // Ensure it doesn't exceed image boundaries
        if bbox.x + bbox.width > max_width {
            bbox.x = max_width - bbox.width;
        }
        if bbox.y + bbox.height > max_height {
            bbox.y = max_height - bbox.height;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_boxes() {
        let mut boxes = vec![Rect::new(10, 10, 50, 50), Rect::new(100, 100, 30, 40)];

        refine_boxes(&mut boxes, 200, 200, 0.1).unwrap();

        // First box should be expanded and square
        assert_eq!(boxes[0].width, boxes[0].height);
        assert!(boxes[0].width > 50);

        // Second box should also be square
        assert_eq!(boxes[1].width, boxes[1].height);
    }

    #[test]
    fn test_refine_boxes_empty() {
        let mut boxes = vec![];

        // Should handle empty input without panic
        refine_boxes(&mut boxes, 200, 200, 0.1).unwrap();
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_refine_boxes_edge_boundaries() {
        let mut boxes = vec![
            // Box at edge of image
            Rect::new(190, 190, 20, 20),
            // Box that would exceed boundaries after expansion
            Rect::new(0, 0, 10, 10),
        ];

        refine_boxes(&mut boxes, 200, 200, 0.5).unwrap();

        // Boxes should not exceed image boundaries
        for bbox in &boxes {
            assert!(bbox.x >= 0);
            assert!(bbox.y >= 0);
            assert!(bbox.x + bbox.width <= 200);
            assert!(bbox.y + bbox.height <= 200);
            // Should still be square
            assert_eq!(bbox.width, bbox.height);
        }
    }

    #[test]
    fn test_refine_boxes_negative_shift() {
        let mut boxes = vec![Rect::new(50, 50, 40, 40)];

        // Negative shift should still work (though unusual)
        refine_boxes(&mut boxes, 200, 200, -0.1).unwrap();

        // Box should be contracted but still valid
        assert!(boxes[0].width > 0);
        assert!(boxes[0].height > 0);
        assert_eq!(boxes[0].width, boxes[0].height);
    }
}
