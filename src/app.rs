//! Main application module for head pose estimation.

use crate::{
    cursor_control::CursorController,
    error::Result,
    face_detection::{FaceDetector, FaceDetection},
    filters::{create_filter, CursorFilter},
    mark_detection::MarkDetector,
    movement_detector::MovementDetector,
    pose_estimation::PoseEstimator,
    utils::refine_boxes,
};
use log::{info, warn};
use opencv::{
    core::{Mat, Point, Scalar, Vec3d, CV_8UC3},
    highgui::{self, WINDOW_NORMAL},
    imgproc::{self, FONT_HERSHEY_SIMPLEX, LINE_8},
    prelude::*,
    videoio::{self, VideoCapture, CAP_PROP_BUFFERSIZE},
};
use std::time::{Duration, Instant};

/// Main application configuration
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Camera index or video file path
    pub video_source: VideoSource,
    /// Cursor control mode
    pub cursor_mode: CursorMode,
    /// Filter type for cursor smoothing
    pub filter_type: String,
    /// Show all filters for comparison
    pub show_all_filters: bool,
    /// GUI display mode
    pub gui_mode: GuiMode,
    /// Image inversion mode
    pub invert_mode: InvertMode,
    /// Brightness adjustment value
    pub brightness: f32,
    /// Enable debug output
    pub debug: bool,
    /// Data source for cursor control
    pub data_source: DataSource,
    /// Vector interpretation mode
    pub vector_mode: VectorMode,
    /// Only move cursor when head movement is detected
    pub cursor_still: bool,
    /// Use relative cursor control
    pub cursor_relative: bool,
}

/// Video source type
#[derive(Debug, Clone)]
pub enum VideoSource {
    /// Webcam index
    Camera(i32),
    /// Video file path
    File(String),
}

/// Cursor control mode
#[derive(Debug, Clone, PartialEq)]
pub enum CursorMode {
    /// No cursor control
    None,
    /// Absolute cursor positioning
    Absolute(String), // filter name
}

/// GUI display mode
#[derive(Debug, Clone, PartialEq)]
pub enum GuiMode {
    /// Show all windows
    All,
    /// Show cursor window only
    Pointers,
    /// Show camera window only
    Camera,
    /// No GUI (headless)
    None,
}

/// Image inversion mode
#[derive(Debug, Clone, PartialEq)]
pub enum InvertMode {
    /// No inversion
    None,
    /// Mirror horizontally
    X,
    /// Flip vertically
    Y,
    /// Both horizontal and vertical
    XY,
}

/// Data source for cursor control
#[derive(Debug, Clone, PartialEq)]
pub enum DataSource {
    /// Use pitch and yaw angles
    PitchYaw,
    /// Use face normal projection
    NormalProjection,
}

/// Vector interpretation mode
#[derive(Debug, Clone, PartialEq)]
pub enum VectorMode {
    /// Direct position mapping
    Location,
    /// Velocity-based control
    Speed,
}

/// Main application struct
pub struct HeadPoseApp {
    config: AppConfig,
    face_detector: FaceDetector,
    mark_detector: MarkDetector,
    pose_estimator: PoseEstimator,
    movement_detector: Option<MovementDetector>,
    cursor_filter: Option<Box<dyn CursorFilter>>,
    video_capture: VideoCapture,
    is_moving: bool,
    cursor_controller: Option<CursorController>,
    all_filters: Option<Vec<(String, Box<dyn CursorFilter>, Scalar)>>,
}

impl HeadPoseApp {
    /// Create a new head pose estimation application
    pub fn new(config: AppConfig) -> Result<Self> {
        info!("Initializing Head Pose Estimation application");

        // Initialize video capture
        let mut video_capture = match &config.video_source {
            VideoSource::Camera(index) => {
                info!("Opening camera {}", index);
                let mut cap = VideoCapture::new(*index, videoio::CAP_ANY)?;
                
                // Reduce buffer size for lower latency (webcam only)
                cap.set(CAP_PROP_BUFFERSIZE, 1.0)?;
                info!("Camera buffer size set to 1 for low latency");
                
                cap
            }
            VideoSource::File(path) => {
                info!("Opening video file: {}", path);
                VideoCapture::from_file(path, videoio::CAP_ANY)?
            }
        };

        // Initialize components
        let face_detector = FaceDetector::new("assets/face_detector.onnx", 0.5, 0.4)?;
        
        let mark_detector = MarkDetector::new("assets/face_landmarks.onnx")?;
        
        // Get initial frame size for camera matrix initialization
        let mut temp_frame = Mat::default();
        video_capture.read(&mut temp_frame)?;
        let frame_width = temp_frame.cols();
        let frame_height = temp_frame.rows();
        
        let pose_estimator = PoseEstimator::new("assets/model.txt", frame_width, frame_height)?;

        // Initialize movement detector if needed
        let movement_detector = if config.cursor_still {
            info!("Movement detection enabled");
            Some(MovementDetector::new(15, 2.0))
        } else {
            None
        };
        
        // Initialize all filters if show_all_filters is enabled
        let all_filters = if config.show_all_filters {
            info!("Showing all filters for comparison");
            let filter_configs = vec![
                ("none", Scalar::new(255.0, 255.0, 255.0, 0.0)),      // White
                ("kalman", Scalar::new(0.0, 255.0, 0.0, 0.0)),        // Green
                ("median", Scalar::new(255.0, 0.0, 0.0, 0.0)),        // Blue
                ("moving_average", Scalar::new(0.0, 255.0, 255.0, 0.0)), // Yellow
                ("exponential", Scalar::new(255.0, 0.0, 255.0, 0.0)), // Magenta
                ("lowpass", Scalar::new(0.0, 165.0, 255.0, 0.0)),     // Orange
                ("lowpass2", Scalar::new(255.0, 100.0, 100.0, 0.0)),  // Light Blue
                ("hampel", Scalar::new(100.0, 255.0, 100.0, 0.0)),    // Light Green
            ];
            
            let mut filters = Vec::new();
            for (name, color) in filter_configs {
                match create_filter(name) {
                    Ok(filter) => filters.push((name.to_string(), filter, color)),
                    Err(e) => warn!("Failed to create {} filter: {}", name, e),
                }
            }
            
            Some(filters)
        } else {
            None
        };

        // Initialize cursor filter and controller
        let (cursor_filter, cursor_controller) = match &config.cursor_mode {
            CursorMode::None => (None, None),
            CursorMode::Absolute(filter_name) => {
                info!("Cursor control enabled with {} filter", filter_name);
                let filter = Some(create_filter(filter_name)?);
                let controller = match CursorController::new() {
                    Ok(c) => {
                        info!("X11 cursor control initialized");
                        Some(c)
                    }
                    Err(e) => {
                        warn!("Failed to initialize cursor control: {}", e);
                        None
                    }
                };
                (filter, controller)
            }
        };

        // Create GUI windows if needed
        if config.gui_mode != GuiMode::None {
            if config.gui_mode == GuiMode::All || config.gui_mode == GuiMode::Camera {
                highgui::named_window("Head Pose Estimation", WINDOW_NORMAL)?;
            }
            if config.gui_mode == GuiMode::All || config.gui_mode == GuiMode::Pointers {
                highgui::named_window("Head Pose Cursor", WINDOW_NORMAL)?;
                highgui::resize_window("Head Pose Cursor", 800, 600)?;
            }
        }

        Ok(Self {
            config,
            face_detector,
            mark_detector,
            pose_estimator,
            movement_detector,
            cursor_filter,
            video_capture,
            is_moving: false,
            cursor_controller,
            all_filters,
        })
    }

    /// Run the main application loop
    pub fn run(&mut self) -> Result<()> {
        info!("Starting main application loop");
        
        let mut frame_count = 0;
        let start_time = Instant::now();
        let mut last_fps_update = Instant::now();
        let mut fps = 0.0;

        info!("Entering main loop");
        loop {
            // Read frame from video source
            let mut frame = Mat::default();
            info!("Reading frame {}", frame_count);
            if !self.video_capture.read(&mut frame)? || frame.empty() {
                if matches!(self.config.video_source, VideoSource::File(_)) {
                    info!("End of video file reached");
                    break;
                }
                warn!("Failed to read frame, retrying...");
                continue;
            }

            // Apply image transformations
            self.apply_transformations(&mut frame)?;

            // Process frame
            let result = self.process_frame(&frame)?;

            // Update FPS counter
            frame_count += 1;
            if last_fps_update.elapsed() >= Duration::from_secs(1) {
                fps = frame_count as f64 / start_time.elapsed().as_secs_f64();
                last_fps_update = Instant::now();
            }

            // Display results
            if self.config.gui_mode != GuiMode::None {
                self.display_results(&frame, &result, fps)?;
                
                // Check for exit
                let key = highgui::wait_key(1)?;
                if key == 27 || key == b'q' as i32 {
                    info!("Exit requested by user");
                    break;
                }
            }
        }

        info!("Application shutting down");
        Ok(())
    }

    /// Apply image transformations (brightness, inversion)
    fn apply_transformations(&self, frame: &mut Mat) -> Result<()> {
        // Apply brightness adjustment
        if self.config.brightness != 0.0 {
            let brightness_scalar = Scalar::new(
                self.config.brightness as f64,
                self.config.brightness as f64,
                self.config.brightness as f64,
                0.0,
            );
            let temp = frame.clone();
            opencv::core::add(&temp, &brightness_scalar, frame, &Mat::default(), -1)?;
        }

        // Apply inversion
        match self.config.invert_mode {
            InvertMode::None => {}
            InvertMode::X => {
                let temp = frame.clone();
                opencv::core::flip(&temp, frame, 1)?;
            }
            InvertMode::Y => {
                let temp = frame.clone();
                opencv::core::flip(&temp, frame, 0)?;
            }
            InvertMode::XY => {
                let temp = frame.clone();
                opencv::core::flip(&temp, frame, -1)?;
            }
        }

        Ok(())
    }

    /// Process a single frame
    fn process_frame(&mut self, frame: &Mat) -> Result<ProcessingResult> {
        // Detect faces
        let faces = self.face_detector.detect(frame)?;
        
        if faces.is_empty() {
            return Ok(ProcessingResult {
                faces: Vec::new(),
                landmarks: Vec::new(),
                poses: Vec::new(),
                cursor_pos: None,
                all_cursor_positions: None,
            });
        }

        // Process each face
        let mut landmarks = Vec::new();
        let mut poses = Vec::new();
        
        for face in &faces {
            // Refine face box
            let mut refined_boxes = vec![face.bbox];
            refine_boxes(
                &mut refined_boxes,
                frame.cols(),
                frame.rows(),
                0.2,
            )?;
            let refined_box = refined_boxes[0];

            // Detect landmarks
            let face_roi = Mat::roi(frame, refined_box)?;
            let face_roi_mat = face_roi.try_clone()?;
            let marks = self.mark_detector.detect(&face_roi_mat)?;
            
            if marks.len() == 68 {
                // Convert Point2f to tuples
                let marks_tuples: Vec<(f32, f32)> = marks.iter()
                    .map(|p| (p.x, p.y))
                    .collect();
                    
                // Estimate pose
                let (rotation_vec, translation_vec, rotation_matrix) = self.pose_estimator.estimate_pose(&marks_tuples)?;
                
                // Extract Euler angles from rotation matrix
                let euler_angles = PoseEstimator::rotation_matrix_to_euler(&rotation_matrix)?;
                let pitch = euler_angles[0];
                let yaw = euler_angles[1];
                let roll = euler_angles[2];
                
                landmarks.push(marks);
                poses.push(PoseData {
                    rotation_vec,
                    translation_vec,
                    rotation_matrix,
                    pitch,
                    yaw,
                    roll,
                });
            }
        }

        // Calculate cursor position if enabled
        let cursor_pos = if !poses.is_empty() && self.cursor_filter.is_some() {
            self.calculate_cursor_position(&poses[0])?
        } else {
            None
        };

        // Calculate all filter positions if enabled
        let all_cursor_positions = if self.all_filters.is_some() && !poses.is_empty() {
            self.calculate_all_cursor_positions(&poses[0])?
        } else {
            None
        };
        
        Ok(ProcessingResult {
            faces,
            landmarks,
            poses,
            cursor_pos,
            all_cursor_positions,
        })
    }

    /// Calculate cursor position based on pose data
    fn calculate_cursor_position(&mut self, pose: &PoseData) -> Result<Option<(f64, f64)>> {
        // Update movement detector if available
        if let Some(detector) = &mut self.movement_detector {
            self.is_moving = detector.update(pose.pitch, pose.yaw);
        }
        let (raw_x, raw_y) = match self.config.data_source {
            DataSource::PitchYaw => {
                // Map angles to normalized coordinates
                let x = pose.yaw / 20.0;  // -10 to 10 degrees -> -0.5 to 0.5
                let y = -pose.pitch / 20.0; // Inverted
                (x, y)
            }
            DataSource::NormalProjection => {
                // TODO: Implement normal projection
                (0.0, 0.0)
            }
        };

        // Apply filter if available
        let (cursor_x, cursor_y) = if let Some(filter) = &mut self.cursor_filter {
            filter.apply(raw_x, raw_y)
        } else {
            (raw_x, raw_y)
        };
        
        // Move cursor if controller is available and conditions are met
        if let Some(controller) = &self.cursor_controller {
            // Check movement condition if cursor_still is enabled
            if !self.config.cursor_still || self.is_moving {
                // Map normalized coordinates to screen coordinates
                let (screen_width, screen_height) = controller.get_screen_size();
                
                if self.config.cursor_relative {
                    // Relative mode - use cursor values as velocity
                    let scale = 5.0; // Sensitivity factor
                    let dx = (cursor_x * scale) as i16;
                    let dy = (cursor_y * scale) as i16;
                    
                    if dx != 0 || dy != 0 {
                        controller.move_relative(dx, dy)?;
                    }
                } else {
                    // Absolute mode - map to screen position
                    // Transform from [-0.5, 0.5] to [0, 1]
                    let norm_x = (cursor_x + 0.5).clamp(0.0, 1.0);
                    let norm_y = (cursor_y + 0.5).clamp(0.0, 1.0);
                    
                    let screen_x = (norm_x * screen_width as f64) as i16;
                    let screen_y = (norm_y * screen_height as f64) as i16;
                    
                    controller.set_position(screen_x, screen_y)?;
                }
            }
        }
        
        Ok(Some((cursor_x, cursor_y)))
    }

    /// Calculate cursor positions for all filters
    fn calculate_all_cursor_positions(&mut self, pose: &PoseData) -> Result<Option<Vec<(String, (f64, f64), Scalar)>>> {
        if let Some(filters) = &mut self.all_filters {
            let (raw_x, raw_y) = match self.config.data_source {
                DataSource::PitchYaw => {
                    let x = pose.yaw / 20.0;
                    let y = -pose.pitch / 20.0;
                    (x, y)
                }
                DataSource::NormalProjection => {
                    // TODO: Implement normal projection
                    (0.0, 0.0)
                }
            };
            
            let mut positions = Vec::new();
            for (name, filter, color) in filters.iter_mut() {
                let (x, y) = filter.apply(raw_x, raw_y);
                positions.push((name.clone(), (x, y), *color));
            }
            
            Ok(Some(positions))
        } else {
            Ok(None)
        }
    }

    /// Display results in GUI windows
    fn display_results(
        &self,
        frame: &Mat,
        result: &ProcessingResult,
        fps: f64,
    ) -> Result<()> {
        // Camera window
        if self.config.gui_mode == GuiMode::All || self.config.gui_mode == GuiMode::Camera {
            let mut display_frame = frame.clone();
            
            // Draw face boxes and landmarks
            for (i, face) in result.faces.iter().enumerate() {
                imgproc::rectangle(
                    &mut display_frame,
                    face.bbox,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    LINE_8,
                    0,
                )?;
                
                // Draw landmarks if available
                if i < result.landmarks.len() {
                    // Get the refined box used for landmark detection
                    let mut refined_boxes = vec![face.bbox];
                    refine_boxes(
                        &mut refined_boxes,
                        display_frame.cols(),
                        display_frame.rows(),
                        0.2,
                    )?;
                    let refined_box = refined_boxes[0];
                    
                    for landmark in &result.landmarks[i] {
                        // Transform landmark from face ROI coordinates to frame coordinates
                        let x = refined_box.x + (landmark.x * refined_box.width as f32 / 256.0) as i32;
                        let y = refined_box.y + (landmark.y * refined_box.height as f32 / 256.0) as i32;
                        
                        imgproc::circle(
                            &mut display_frame,
                            Point::new(x, y),
                            2,
                            Scalar::new(255.0, 0.0, 0.0, 0.0),
                            -1,
                            LINE_8,
                            0,
                        )?;
                    }
                }
                
                // Draw pose information if available
                if i < result.poses.len() {
                    let pose = &result.poses[i];
                    let pose_text = format!("Pitch: {:.1} Yaw: {:.1} Roll: {:.1}", 
                        pose.pitch, pose.yaw, pose.roll);
                    imgproc::put_text(
                        &mut display_frame,
                        &pose_text,
                        Point::new(face.bbox.x, face.bbox.y - 10),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar::new(0.0, 255.0, 255.0, 0.0),
                        1,
                        LINE_8,
                        false,
                    )?;
                    
                    // Transform landmarks to frame coordinates for pose axes
                    let mut refined_boxes = vec![face.bbox];
                    refine_boxes(
                        &mut refined_boxes,
                        display_frame.cols(),
                        display_frame.rows(),
                        0.2,
                    )?;
                    let refined_box = refined_boxes[0];
                    
                    let transformed_landmarks: Vec<opencv::core::Point2f> = result.landmarks[i].iter()
                        .map(|lm| opencv::core::Point2f::new(
                            refined_box.x as f32 + (lm.x * refined_box.width as f32 / 256.0),
                            refined_box.y as f32 + (lm.y * refined_box.height as f32 / 256.0),
                        ))
                        .collect();
                    
                    // Draw pose axes (simplified visualization using rotation matrix)
                    self.draw_pose_axes(&mut display_frame, &transformed_landmarks, &pose.rotation_matrix, &pose.translation_vec)?;
                }
            }

            // Draw FPS
            let fps_text = format!("FPS: {:.1}", fps);
            imgproc::put_text(
                &mut display_frame,
                &fps_text,
                Point::new(10, 30),
                FONT_HERSHEY_SIMPLEX,
                1.0,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                LINE_8,
                false,
            )?;

            // Draw movement detection status if available
            if self.movement_detector.is_some() && self.is_moving {
                imgproc::put_text(
                        &mut display_frame,
                        "MOVING",
                        Point::new(10, 60),
                        FONT_HERSHEY_SIMPLEX,
                        1.0,
                        Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        LINE_8,
                        false,
                    )?;
            }
            
            highgui::imshow("Head Pose Estimation", &display_frame)?;
        }

        // Cursor window
        if self.config.gui_mode == GuiMode::All || self.config.gui_mode == GuiMode::Pointers {
            let mut cursor_frame = Mat::zeros(600, 800, CV_8UC3)?.to_mat()?;
            
            // Draw all filter positions if available
            if let Some(all_positions) = &result.all_cursor_positions {
                // Draw legend
                let mut legend_y = 20;
                for (name, _, color) in all_positions {
                    imgproc::put_text(
                        &mut cursor_frame,
                        name,
                        Point::new(10, legend_y),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        *color,
                        1,
                        LINE_8,
                        false,
                    )?;
                    legend_y += 20;
                }
                
                // Draw cursor positions
                for (_name, (x, y), color) in all_positions {
                    let cursor_x = ((x + 0.5) * 800.0) as i32;
                    let cursor_y = ((y + 0.5) * 600.0) as i32;
                    
                    // Draw circle with alpha blending for overlapping cursors
                    imgproc::circle(
                        &mut cursor_frame,
                        Point::new(cursor_x, cursor_y),
                        8,
                        *color,
                        2,
                        LINE_8,
                        0,
                    )?;
                    
                    // Draw filled circle with transparency effect
                    imgproc::circle(
                        &mut cursor_frame,
                        Point::new(cursor_x, cursor_y),
                        6,
                        Scalar::new(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 0.0),
                        -1,
                        LINE_8,
                        0,
                    )?;
                }
            } else if let Some((x, y)) = result.cursor_pos {
                // Single cursor mode
                let cursor_x = ((x + 0.5) * 800.0) as i32;
                let cursor_y = ((y + 0.5) * 600.0) as i32;
                
                imgproc::circle(
                    &mut cursor_frame,
                    Point::new(cursor_x, cursor_y),
                    10,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    -1,
                    LINE_8,
                    0,
                )?;
            }

            highgui::imshow("Head Pose Cursor", &cursor_frame)?;
        }

        Ok(())
    }
    
    /// Draw pose axes on the frame
    fn draw_pose_axes(
        &self,
        frame: &mut Mat,
        landmarks: &[opencv::core::Point2f],
        rotation_mat: &Mat,
        _translation_vec: &Vec3d,
    ) -> Result<()> {
        // Use nose tip as origin (landmark 30)
        if landmarks.len() > 30 {
            let nose_tip = &landmarks[30];
            let origin = Point::new(nose_tip.x as i32, nose_tip.y as i32);
            
            // Define axis lengths
            let axis_length = 50.0;
            
            // Get rotation matrix values
            let r11 = *rotation_mat.at_2d::<f64>(0, 0)?;
            let r12 = *rotation_mat.at_2d::<f64>(0, 1)?;
            let r21 = *rotation_mat.at_2d::<f64>(1, 0)?;
            let r22 = *rotation_mat.at_2d::<f64>(1, 1)?;
            let r31 = *rotation_mat.at_2d::<f64>(2, 0)?;
            let r32 = *rotation_mat.at_2d::<f64>(2, 1)?;
            
            // Project 3D axes to 2D
            // X-axis (red)
            let x_end = Point::new(
                origin.x + (r11 * axis_length) as i32,
                origin.y + (r21 * axis_length) as i32,
            );
            imgproc::arrowed_line(
                frame,
                origin,
                x_end,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                LINE_8,
                0,
                0.2,
            )?;
            
            // Y-axis (green)
            let y_end = Point::new(
                origin.x + (r12 * axis_length) as i32,
                origin.y + (r22 * axis_length) as i32,
            );
            imgproc::arrowed_line(
                frame,
                origin,
                y_end,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                LINE_8,
                0,
                0.2,
            )?;
            
            // Z-axis (blue) - pointing out of the face
            let z_end = Point::new(
                origin.x + (r31 * axis_length) as i32,
                origin.y + (r32 * axis_length) as i32,
            );
            imgproc::arrowed_line(
                frame,
                origin,
                z_end,
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                LINE_8,
                0,
                0.2,
            )?;
        }
        
        Ok(())
    }
}

/// Result of processing a single frame
struct ProcessingResult {
    faces: Vec<FaceDetection>,
    landmarks: Vec<Vec<opencv::core::Point2f>>,
    poses: Vec<PoseData>,
    cursor_pos: Option<(f64, f64)>,
    all_cursor_positions: Option<Vec<(String, (f64, f64), Scalar)>>,
}

/// Pose estimation data
struct PoseData {
    #[allow(dead_code)]
    rotation_vec: Vec3d,
    #[allow(dead_code)]
    translation_vec: Vec3d,
    rotation_matrix: Mat,
    pitch: f64,
    yaw: f64,
    #[allow(dead_code)]
    roll: f64,
}