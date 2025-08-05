//! Head pose estimation application for real-time tracking and cursor control.

use anyhow::Result;
use clap::Parser;
use head_pose_estimation::app::{
    AppConfig, CursorConfig, CursorMode, DataSource, DisplayConfig, GuiMode, HeadPoseApp, InvertMode, VectorMode, VideoSource,
};
use log::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Camera index to use
    #[arg(long, default_value = "0")]
    cam: i32,

    /// Video file to process
    #[arg(short, long)]
    video: Option<String>,

    /// Cursor control mode (none, or filter name for absolute control)
    #[arg(short = 'c', long = "cursor")]
    cursor: Option<String>,

    /// Filter type for display (when cursor is not used)
    #[arg(short, long, default_value = "kalman")]
    filter: String,
    
    /// Show all filters for comparison
    #[arg(long, default_value = "false")]
    cursor_filter_all: bool,

    /// GUI display mode (all, pointers, cam, none)
    #[arg(short, long, default_value = "all")]
    gui: String,

    /// Invert image (none, x, y, xy)
    #[arg(short, long, default_value = "none")]
    inv: String,

    /// Brightness adjustment value (0 to disable, typical: 30)
    #[arg(short, long, default_value = "0")]
    brightness: f32,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Data source for cursor control (pitchyaw, normalproj)
    #[arg(long, default_value = "pitchyaw")]
    datasource: String,

    /// Vector interpretation mode (location, speed)
    #[arg(long, default_value = "location")]
    vector: String,

    /// Only move cursor when head movement is detected
    #[arg(long)]
    cursor_still: bool,

    /// Use relative cursor control (hold 'w' key to activate)
    #[arg(long)]
    cursor_relative: bool,

    /// Path to configuration file (YAML format)
    #[arg(short = 'C', long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logger
    if args.debug {
        env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));
    } else {
        env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    }

    info!("Head Pose Estimation - Rust Port");

    // Load configuration if provided
    let mut config_file = if let Some(config_path) = &args.config {
        info!("Loading configuration from: {}", config_path);
        match head_pose_estimation::config::Config::from_file(config_path) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                log::warn!("Failed to load config file: {}. Using defaults.", e);
                None
            }
        }
    } else {
        None
    };

    // Build application configuration
    let config = AppConfig {
        video_source: if let Some(video_path) = args.video {
            VideoSource::File(video_path)
        } else {
            VideoSource::Camera(args.cam)
        },
        cursor_mode: if let Some(filter) = args.cursor {
            CursorMode::Absolute(filter)
        } else {
            CursorMode::None
        },
        filter_type: args.filter,
        display: DisplayConfig::new(args.cursor_filter_all, args.debug),
        gui_mode: match args.gui.as_str() {
            "all" => GuiMode::All,
            "pointers" => GuiMode::Pointers,
            "cam" => GuiMode::Camera,
            "none" => GuiMode::None,
            _ => GuiMode::All,
        },
        invert_mode: match args.inv.as_str() {
            "x" => InvertMode::X,
            "y" => InvertMode::Y,
            "xy" => InvertMode::XY,
            _ => InvertMode::None,
        },
        brightness: args.brightness,
        data_source: match args.datasource.as_str() {
            "normalproj" => DataSource::NormalProjection,
            _ => DataSource::PitchYaw,
        },
        vector_mode: match args.vector.as_str() {
            "speed" => VectorMode::Speed,
            _ => VectorMode::Location,
        },
        cursor_config: CursorConfig::new(args.cursor_still, args.cursor_relative),
    };

    // Create and run application
    let mut app = HeadPoseApp::new(config)?;
    app.run()?;

    Ok(())
}
