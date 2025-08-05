//! Main application module for head pose estimation.

use crate::{
    cursor_control::CursorController,
    error::Result,
    face_detection::{FaceDetection, FaceDetector},
    filters::{create_filter, CursorFilter},
    mark_detection::MarkDetector,
    movement_detector::MovementDetector,
    pose_estimation::PoseEstimator,
    utils::refine_boxes,
    utils::safe_cast::{f32_to_i32_clamp, f64_to_i32},
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

/// Type alias for filter collection: (name, filter, color)
type FilterCollection = Vec<(String, Box<dyn CursorFilter>, Scalar)>;

/// Type alias for cursor position collection: (name, position, color)
type CursorPositions = Vec<(String, (f64, f64), Scalar)>;

/// Cursor behavior configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorConfig {
    /// Movement threshold behavior
    pub movement_threshold: bool,
    /// Coordinate mode (absolute vs relative)
    pub relative_mode: bool,
}

impl CursorConfig {
    /// Create new cursor configuration
    #[must_use]
    pub const fn new(movement_threshold: bool, relative_mode: bool) -> Self {
        Self {
            movement_threshold,
            relative_mode,
        }
    }
}

/// Display options configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayConfig {
    /// Show all filters for comparison
    pub show_all_filters: bool,
    /// Enable debug output
    pub debug: bool,
}

impl DisplayConfig {
    /// Create new display configuration
    #[must_use]
    pub const fn new(show_all_filters: bool, debug: bool) -> Self {
        Self {
            show_all_filters,
            debug,
        }
    }
}

/// Main application configuration
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Camera index or video file path
    pub video_source: VideoSource,
    /// Cursor control mode
    pub cursor_mode: CursorMode,
    /// Filter type for cursor smoothing
    pub filter_type: String,
    /// Display configuration
    pub display: DisplayConfig,
    /// GUI display mode
    pub gui_mode: GuiMode,
    /// Image inversion mode
    pub invert_mode: InvertMode,
    /// Brightness adjustment value
    pub brightness: f32,
    /// Data source for cursor control
    pub data_source: DataSource,
    /// Vector interpretation mode
    pub vector_mode: VectorMode,
    /// Cursor behavior configuration
    pub cursor_config: CursorConfig,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CursorMode {
    /// No cursor control
    None,
    /// Absolute cursor positioning
    Absolute(String), // filter name
}

/// GUI display mode
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSource {
    /// Use pitch and yaw angles
    PitchYaw,
    /// Use face normal projection
    NormalProjection,
}

/// Vector interpretation mode
#[derive(Debug, Clone, PartialEq, Eq)]
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
    all_filters: Option<FilterCollection>,
}

impl HeadPoseApp {
    /// Create a new head pose estimation application
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Video capture initialization fails
    /// - Model loading fails
    /// - Filter creation fails
    /// - Cursor controller initialization fails
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
                let cap = VideoCapture::from_file(path, videoio::CAP_ANY)?;

                // Check if the video file was opened successfully
                if !cap.is_opened()? {
                    return Err(crate::error::Error::ModelError(format!(
                        "Failed to open video file: {}",
                        path
                    )));
                }

                cap
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
        let movement_detector = if config.cursor_config.movement_threshold {
            info!("Movement detection enabled");
            Some(MovementDetector::new(15, 2.0))
        } else {
            None
        };

        // Initialize all filters if show_all_filters is enabled
        let all_filters = if config.display.show_all_filters {
            info!("Showing all filters for comparison");
            let filter_configs = vec![
                ("none", Scalar::new(255.0, 255.0, 255.0, 0.0)),         // White
                ("kalman", Scalar::new(0.0, 255.0, 0.0, 0.0)),           // Green
                ("median", Scalar::new(255.0, 0.0, 0.0, 0.0)),           // Blue
                ("moving_average", Scalar::new(0.0, 255.0, 255.0, 0.0)), // Yellow
                ("exponential", Scalar::new(255.0, 0.0, 255.0, 0.0)),    // Magenta
                ("lowpass", Scalar::new(0.0, 165.0, 255.0, 0.0)),        // Orange
                ("lowpass2", Scalar::new(255.0, 100.0, 100.0, 0.0)),     // Light Blue
                ("hampel", Scalar::new(100.0, 255.0, 100.0, 0.0)),       // Light Green
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
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame capture fails
    /// - Face/landmark detection fails
    /// - Pose estimation fails
    /// - OpenCV operations fail
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
                fps = f64::from(frame_count) / start_time.elapsed().as_secs_f64();
                last_fps_update = Instant::now();
            }

            // Display results
            if self.config.gui_mode != GuiMode::None {
                self.display_results(&frame, &result, fps)?;

                // Check for exit
                let key = highgui::wait_key(1)?;
                if key == 27 || key == i32::from(b'q') {
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
                f64::from(self.config.brightness),
                f64::from(self.config.brightness),
                f64::from(self.config.brightness),
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
            refine_boxes(&mut refined_boxes, frame.cols(), frame.rows(), 0.2)?;
            let refined_box = refined_boxes[0];

            // Detect landmarks
            let face_roi = Mat::roi(frame, refined_box)?;
            let face_roi_mat = face_roi.try_clone()?;
            let marks = self.mark_detector.detect(&face_roi_mat)?;

            if marks.len() == 68 {
                // Convert Point2f to tuples
                let marks_tuples: Vec<(f32, f32)> = marks.iter().map(|p| (p.x, p.y)).collect();

                // Estimate pose
                let (rotation_vec, translation_vec, rotation_matrix) =
                    self.pose_estimator.estimate_pose(&marks_tuples)?;

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
            self.calculate_all_cursor_positions(&poses[0])
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
                let x = pose.yaw / 20.0; // -10 to 10 degrees -> -0.5 to 0.5
                let y = -pose.pitch / 20.0; // Inverted
                (x, y)
            }
            DataSource::NormalProjection => {
                // Calculate normal projection from rotation matrix
                // The normal vector is the third column of the rotation matrix
                let r13 = *pose.rotation_matrix.at_2d::<f64>(0, 2).unwrap_or(&0.0);
                let r23 = *pose.rotation_matrix.at_2d::<f64>(1, 2).unwrap_or(&0.0);

                // Scale the projection for cursor movement
                // Invert Y to match screen coordinates
                let scale = 100.0;
                (r13 * scale, -r23 * scale)
            }
        };

        // Apply filter if available
        let (cursor_x, cursor_y) = self
            .cursor_filter
            .as_mut()
            .map_or((raw_x, raw_y), |filter| filter.apply(raw_x, raw_y));

        // Move cursor if controller is available and conditions are met
        if let Some(controller) = &self.cursor_controller {
            // Check movement condition if movement threshold is enabled
            if !self.config.cursor_config.movement_threshold || self.is_moving {
                // Map normalized coordinates to screen coordinates
                let (screen_width, screen_height) = controller.get_screen_size();

                if self.config.cursor_config.relative_mode {
                    // Relative mode - use cursor values as velocity
                    let scale = 5.0; // Sensitivity factor
                    let dx = f64_to_i32(cursor_x * scale)
                        .unwrap_or(0)
                        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                    let dy = f64_to_i32(cursor_y * scale)
                        .unwrap_or(0)
                        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;

                    if dx != 0 || dy != 0 {
                        controller.move_relative(dx, dy)?;
                    }
                } else {
                    // Absolute mode - map to screen position
                    // Transform from [-0.5, 0.5] to [0, 1]
                    let norm_x = (cursor_x + 0.5).clamp(0.0, 1.0);
                    let norm_y = (cursor_y + 0.5).clamp(0.0, 1.0);

                    let screen_x = f64_to_i32(norm_x * f64::from(screen_width))
                        .unwrap_or(0)
                        .clamp(0, i32::from(i16::MAX)) as i16;
                    let screen_y = f64_to_i32(norm_y * f64::from(screen_height))
                        .unwrap_or(0)
                        .clamp(0, i32::from(i16::MAX)) as i16;

                    controller.set_position(screen_x, screen_y)?;
                }
            }
        }

        Ok(Some((cursor_x, cursor_y)))
    }

    /// Calculate cursor positions for all filters
    fn calculate_all_cursor_positions(&mut self, pose: &PoseData) -> Option<CursorPositions> {
        if let Some(filters) = &mut self.all_filters {
            let (raw_x, raw_y) = match self.config.data_source {
                DataSource::PitchYaw => {
                    let x = pose.yaw / 20.0;
                    let y = -pose.pitch / 20.0;
                    (x, y)
                }
                DataSource::NormalProjection => {
                    // Calculate normal projection from rotation matrix
                    let r13 = *pose.rotation_matrix.at_2d::<f64>(0, 2).unwrap_or(&0.0);
                    let r23 = *pose.rotation_matrix.at_2d::<f64>(1, 2).unwrap_or(&0.0);

                    // Scale the projection for cursor movement
                    let scale = 100.0;
                    (r13 * scale, -r23 * scale)
                }
            };

            let mut positions = Vec::new();
            for (name, filter, color) in filters.iter_mut() {
                let (x, y) = filter.apply(raw_x, raw_y);
                positions.push((name.clone(), (x, y), *color));
            }

            Some(positions)
        } else {
            None
        }
    }

    /// Display results in GUI windows
    fn display_results(&self, frame: &Mat, result: &ProcessingResult, fps: f64) -> Result<()> {
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
                    refine_boxes(&mut refined_boxes, display_frame.cols(), display_frame.rows(), 0.2)?;
                    let refined_box = refined_boxes[0];

                    for landmark in &result.landmarks[i] {
                        // Transform landmark from face ROI coordinates to frame coordinates
                        let x = refined_box.x
                            + f32_to_i32_clamp(landmark.x * refined_box.width as f32 / 256.0, 0, i32::MAX);
                        let y = refined_box.y
                            + f32_to_i32_clamp(landmark.y * refined_box.height as f32 / 256.0, 0, i32::MAX);

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
                    let pose_text = format!("Pitch: {:.1} Yaw: {:.1} Roll: {:.1}", pose.pitch, pose.yaw, pose.roll);
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
                    refine_boxes(&mut refined_boxes, display_frame.cols(), display_frame.rows(), 0.2)?;
                    let refined_box = refined_boxes[0];

                    let transformed_landmarks: Vec<opencv::core::Point2f> = result.landmarks[i]
                        .iter()
                        .map(|lm| {
                            opencv::core::Point2f::new(
                                refined_box.x as f32 + (lm.x * refined_box.width as f32 / 256.0),
                                refined_box.y as f32 + (lm.y * refined_box.height as f32 / 256.0),
                            )
                        })
                        .collect();

                    // Draw pose axes (simplified visualization using rotation matrix)
                    Self::draw_pose_axes(
                        &mut display_frame,
                        &transformed_landmarks,
                        &pose.rotation_matrix,
                        &pose.translation_vec,
                    )?;

                    // Draw normal vector when using normal projection
                    if matches!(self.config.data_source, DataSource::NormalProjection) {
                        Self::draw_normal_vector(
                            &mut display_frame,
                            &transformed_landmarks,
                            &pose.rotation_matrix,
                            &pose.translation_vec,
                        )?;
                    }
                }
            }

            // Draw FPS
            let fps_text = format!("FPS: {fps:.1}");
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
                    let cursor_x = f64_to_i32((x + 0.5) * 800.0).unwrap_or(0);
                    let cursor_y = f64_to_i32((y + 0.5) * 600.0).unwrap_or(0);

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
                let cursor_x = f64_to_i32((x + 0.5) * 800.0).unwrap_or(0);
                let cursor_y = f64_to_i32((y + 0.5) * 600.0).unwrap_or(0);

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

            // Draw debug overlay if enabled
            if self.config.display.debug && !result.poses.is_empty() {
                let pose = &result.poses[0];
                let text_color = Scalar::new(200.0, 200.0, 200.0, 0.0);

                // Draw angle values or normal projection values
                match self.config.data_source {
                    DataSource::PitchYaw => {
                        let pitch_text = format!("Pitch: {:.1}", pose.pitch);
                        let yaw_text = format!("Yaw: {:.1}", pose.yaw);
                        let roll_text = format!("Roll: {:.1}", pose.roll);

                        imgproc::put_text(
                            &mut cursor_frame,
                            &pitch_text,
                            Point::new(10, 450),
                            FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                            LINE_8,
                            false,
                        )?;

                        imgproc::put_text(
                            &mut cursor_frame,
                            &yaw_text,
                            Point::new(10, 480),
                            FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                            LINE_8,
                            false,
                        )?;

                        imgproc::put_text(
                            &mut cursor_frame,
                            &roll_text,
                            Point::new(10, 510),
                            FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                            LINE_8,
                            false,
                        )?;
                    }
                    DataSource::NormalProjection => {
                        // Display normal projection values
                        if let Some(pose) = result.poses.first() {
                            let r13 = *pose.rotation_matrix.at_2d::<f64>(0, 2).unwrap_or(&0.0);
                            let r23 = *pose.rotation_matrix.at_2d::<f64>(1, 2).unwrap_or(&0.0);
                            let text = format!("Normal: X={:.2} Y={:.2}", r13, r23);
                            imgproc::put_text(
                                &mut cursor_frame,
                                &text,
                                Point::new(10, 450),
                                FONT_HERSHEY_SIMPLEX,
                                0.7,
                                text_color,
                                2,
                                LINE_8,
                                false,
                            )?;
                        } else {
                            let text = "No face detected";
                            imgproc::put_text(
                                &mut cursor_frame,
                                text,
                                Point::new(10, 450),
                                FONT_HERSHEY_SIMPLEX,
                                0.7,
                                text_color,
                                2,
                                LINE_8,
                                false,
                            )?;
                        }
                    }
                }

                // Draw movement status if detector is active
                if self.movement_detector.is_some() {
                    let status_y = match self.config.data_source {
                        DataSource::PitchYaw => 540,
                        DataSource::NormalProjection => 480,
                    };

                    let (status_text, status_color) = if self.is_moving {
                        ("Status: MOVING", Scalar::new(0.0, 255.0, 0.0, 0.0))
                    } else {
                        ("Status: STILL", Scalar::new(0.0, 100.0, 255.0, 0.0))
                    };

                    imgproc::put_text(
                        &mut cursor_frame,
                        status_text,
                        Point::new(10, status_y),
                        FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                        LINE_8,
                        false,
                    )?;

                    // Draw movement statistics if available
                    if let Some(detector) = &self.movement_detector {
                        if let Some((pitch_stats, yaw_stats)) = detector.get_stats() {
                            let movement_text = format!(
                                "Movement: P_std={:.2} Y_std={:.2}",
                                pitch_stats.std_dev, yaw_stats.std_dev
                            );

                            imgproc::put_text(
                                &mut cursor_frame,
                                &movement_text,
                                Point::new(10, status_y + 30),
                                FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_color,
                                1,
                                LINE_8,
                                false,
                            )?;
                        }
                    }
                }
            }

            highgui::imshow("Head Pose Cursor", &cursor_frame)?;
        }

        Ok(())
    }

    /// Draw pose axes on the frame
    fn draw_pose_axes(
        frame: &mut Mat,
        landmarks: &[opencv::core::Point2f],
        rotation_mat: &Mat,
        _translation_vec: &Vec3d,
    ) -> Result<()> {
        // Use nose tip as origin (landmark 30)
        if landmarks.len() > 30 {
            let nose_tip = &landmarks[30];
            let origin = Point::new(
                f32_to_i32_clamp(nose_tip.x, i32::MIN, i32::MAX),
                f32_to_i32_clamp(nose_tip.y, i32::MIN, i32::MAX),
            );

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
                origin.x + f64_to_i32(r11 * axis_length).unwrap_or(0),
                origin.y + f64_to_i32(r21 * axis_length).unwrap_or(0),
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
                origin.x + f64_to_i32(r12 * axis_length).unwrap_or(0),
                origin.y + f64_to_i32(r22 * axis_length).unwrap_or(0),
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
                origin.x + f64_to_i32(r31 * axis_length).unwrap_or(0),
                origin.y + f64_to_i32(r32 * axis_length).unwrap_or(0),
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

    /// Draw normal vector from the face
    fn draw_normal_vector(
        frame: &mut Mat,
        landmarks: &[opencv::core::Point2f],
        rotation_mat: &Mat,
        _translation_vec: &Vec3d,
    ) -> Result<()> {
        // Use nose tip as origin (landmark 30)
        if landmarks.len() > 30 {
            let nose_tip = &landmarks[30];
            let origin = Point::new(
                f32_to_i32_clamp(nose_tip.x, i32::MIN, i32::MAX),
                f32_to_i32_clamp(nose_tip.y, i32::MIN, i32::MAX),
            );

            // Define normal vector length (longer than axes for visibility)
            let normal_length = 100.0;

            // Get rotation matrix values for Z-axis (normal vector)
            let r13 = *rotation_mat.at_2d::<f64>(0, 2)?;
            let r23 = *rotation_mat.at_2d::<f64>(1, 2)?;

            // Project normal vector to 2D
            // The normal vector points out of the face (negative Z in face coordinate system)
            let normal_end = Point::new(
                origin.x - f64_to_i32(r13 * normal_length).unwrap_or(0),
                origin.y - f64_to_i32(r23 * normal_length).unwrap_or(0),
            );

            // Draw normal vector in cyan color with thicker line
            imgproc::arrowed_line(
                frame,
                origin,
                normal_end,
                Scalar::new(255.0, 255.0, 0.0, 0.0), // Cyan (BGR)
                3,
                LINE_8,
                0,
                0.3,
            )?;

            // Draw a small circle at the origin for clarity
            imgproc::circle(
                frame,
                origin,
                3,
                Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
                -1,
                LINE_8,
                0,
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
    all_cursor_positions: Option<CursorPositions>,
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
