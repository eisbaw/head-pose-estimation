//! Utility functions for image processing and coordinate transformations.

use opencv::core::Rect;
use crate::Result;

/// Refine bounding boxes to ensure they are within image boundaries
/// 
/// This is a port of the Python `refine()` function from utils.py
pub fn refine_boxes(boxes: &mut [Rect], max_width: i32, max_height: i32, shift: f32) -> Result<()> {
    for bbox in boxes.iter_mut() {
        let x_shift = (bbox.width as f32 * shift) as i32;
        let y_shift = (bbox.height as f32 * shift) as i32;
        
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
        let mut boxes = vec![
            Rect::new(10, 10, 50, 50),
            Rect::new(100, 100, 30, 40),
        ];
        
        refine_boxes(&mut boxes, 200, 200, 0.1).unwrap();
        
        // First box should be expanded and square
        assert_eq!(boxes[0].width, boxes[0].height);
        assert!(boxes[0].width > 50);
        
        // Second box should also be square
        assert_eq!(boxes[1].width, boxes[1].height);
    }
}