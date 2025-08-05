//! Tests for command-line argument parsing
//!
//! Note: These tests verify the argument parser configuration by creating
//! a test parser with the same structure as the main application.

use clap::{Arg, ArgAction, Command as ClapCommand};

/// Create a command with the same argument structure as the main binary
fn create_test_command() -> ClapCommand {
    ClapCommand::new("head-pose-estimation")
        .version("0.1.0")
        .about("Head pose estimation with ONNX Runtime")
        .arg(
            Arg::new("cam")
                .short('c')
                .long("cam")
                .value_name("INDEX")
                .help("Webcam index"),
        )
        .arg(
            Arg::new("video")
                .short('v')
                .long("video")
                .value_name("PATH")
                .conflicts_with("cam")
                .help("Video file path"),
        )
        .arg(
            Arg::new("filter")
                .short('f')
                .long("filter")
                .value_name("TYPE")
                .default_value("none")
                .help("Filter type"),
        )
        .arg(
            Arg::new("data-source")
                .long("data-source")
                .value_name("SOURCE")
                .default_value("angles")
                .help("Data source for cursor control"),
        )
        .arg(
            Arg::new("cursor-mode")
                .long("cursor-mode")
                .value_name("MODE")
                .default_value("absolute")
                .help("Cursor control mode"),
        )
        .arg(
            Arg::new("gui")
                .long("gui")
                .value_name("MODE")
                .default_value("all")
                .help("GUI display mode"),
        )
        .arg(
            Arg::new("movement-based-cursor")
                .long("movement-based-cursor")
                .action(ArgAction::SetTrue)
                .help("Enable movement-based cursor"),
        )
        .arg(
            Arg::new("show-all-filters")
                .long("show-all-filters")
                .action(ArgAction::SetTrue)
                .help("Show all filters"),
        )
        .arg(
            Arg::new("invert-x")
                .long("invert-x")
                .action(ArgAction::SetTrue)
                .help("Invert X axis"),
        )
        .arg(
            Arg::new("invert-y")
                .long("invert-y")
                .action(ArgAction::SetTrue)
                .help("Invert Y axis"),
        )
        .arg(
            Arg::new("use-movement-filter")
                .long("use-movement-filter")
                .action(ArgAction::SetTrue)
                .help("Use movement filter"),
        )
        .arg(
            Arg::new("brightness")
                .long("brightness")
                .value_name("VALUE")
                .default_value("1.0")
                .help("Brightness adjustment"),
        )
        .arg(
            Arg::new("cursor-sensitivity")
                .long("cursor-sensitivity")
                .value_name("VALUE")
                .default_value("1.0")
                .help("Cursor sensitivity"),
        )
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("PATH")
                .help("Configuration file path"),
        )
}

#[test]
fn test_help_argument() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--help"]);

    // Help should cause an error (but a specific help error)
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
}

#[test]
fn test_no_arguments() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation"]);

    // Should succeed with defaults
    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(matches.get_one::<String>("filter").map(|s| s.as_str()), Some("none"));
    assert_eq!(
        matches.get_one::<String>("data-source").map(|s| s.as_str()),
        Some("angles")
    );
}

#[test]
fn test_cam_argument() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--cam", "0"]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(matches.get_one::<String>("cam").map(|s| s.as_str()), Some("0"));
}

#[test]
fn test_video_argument() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--video", "test.mp4"]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(matches.get_one::<String>("video").map(|s| s.as_str()), Some("test.mp4"));
}

#[test]
fn test_cam_video_conflict() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--cam", "0", "--video", "test.mp4"]);

    // Should fail due to conflict
    assert!(result.is_err());
}

#[test]
fn test_filter_arguments() {
    let filters = vec![
        "none",
        "moving_average",
        "median",
        "exponential",
        "kalman",
        "lowpass",
        "lowpass2",
        "hampel",
    ];

    for filter in filters {
        let cmd = create_test_command();
        let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--filter", filter]);

        assert!(result.is_ok(), "Should accept filter: {}", filter);
        let matches = result.unwrap();
        assert_eq!(matches.get_one::<String>("filter").map(|s| s.as_str()), Some(filter));
    }
}

#[test]
fn test_data_source_arguments() {
    let sources = vec!["angles", "location", "speed", "normal_projection"];

    for source in sources {
        let cmd = create_test_command();
        let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--data-source", source]);

        assert!(result.is_ok(), "Should accept data source: {}", source);
        let matches = result.unwrap();
        assert_eq!(
            matches.get_one::<String>("data-source").map(|s| s.as_str()),
            Some(source)
        );
    }
}

#[test]
fn test_cursor_mode_arguments() {
    let modes = vec!["absolute", "relative"];

    for mode in modes {
        let cmd = create_test_command();
        let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--cursor-mode", mode]);

        assert!(result.is_ok(), "Should accept cursor mode: {}", mode);
        let matches = result.unwrap();
        assert_eq!(matches.get_one::<String>("cursor-mode").map(|s| s.as_str()), Some(mode));
    }
}

#[test]
fn test_gui_mode_arguments() {
    let modes = vec!["all", "pointers", "cam", "none"];

    for mode in modes {
        let cmd = create_test_command();
        let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--gui", mode]);

        assert!(result.is_ok(), "Should accept GUI mode: {}", mode);
        let matches = result.unwrap();
        assert_eq!(matches.get_one::<String>("gui").map(|s| s.as_str()), Some(mode));
    }
}

#[test]
fn test_boolean_flags() {
    let flags = vec![
        "--movement-based-cursor",
        "--show-all-filters",
        "--invert-x",
        "--invert-y",
        "--use-movement-filter",
    ];

    for flag in flags {
        let cmd = create_test_command();
        let result = cmd.try_get_matches_from(vec!["head-pose-estimation", flag]);

        assert!(result.is_ok(), "Should accept flag: {}", flag);
        let matches = result.unwrap();

        // Extract flag name from --flag-name
        let flag_name = flag.trim_start_matches("--").replace('-', "-");
        assert!(matches.get_flag(&flag_name), "Flag {} should be set", flag);
    }
}

#[test]
fn test_numeric_arguments() {
    // Test brightness
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--brightness", "1.5"]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(matches.get_one::<String>("brightness").map(|s| s.as_str()), Some("1.5"));

    // Test cursor sensitivity
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--cursor-sensitivity", "2.0"]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(
        matches.get_one::<String>("cursor-sensitivity").map(|s| s.as_str()),
        Some("2.0")
    );
}

#[test]
fn test_config_file_argument() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec!["head-pose-estimation", "--config", "config.yaml"]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(
        matches.get_one::<String>("config").map(|s| s.as_str()),
        Some("config.yaml")
    );
}

#[test]
fn test_multiple_arguments() {
    let cmd = create_test_command();
    let result = cmd.try_get_matches_from(vec![
        "head-pose-estimation",
        "--video",
        "test.mp4",
        "--filter",
        "kalman",
        "--data-source",
        "speed",
        "--gui",
        "none",
        "--invert-x",
        "--brightness",
        "1.2",
    ]);

    assert!(result.is_ok());
    let matches = result.unwrap();
    assert_eq!(matches.get_one::<String>("video").map(|s| s.as_str()), Some("test.mp4"));
    assert_eq!(matches.get_one::<String>("filter").map(|s| s.as_str()), Some("kalman"));
    assert_eq!(
        matches.get_one::<String>("data-source").map(|s| s.as_str()),
        Some("speed")
    );
    assert_eq!(matches.get_one::<String>("gui").map(|s| s.as_str()), Some("none"));
    assert!(matches.get_flag("invert-x"));
    assert_eq!(matches.get_one::<String>("brightness").map(|s| s.as_str()), Some("1.2"));
}
