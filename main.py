"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import numpy as np
import subprocess
import re
import time
import threading

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
from cursor_filters import create_cursor_filter, CursorFilter
from movement_detector import MovementDetector

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
parser.add_argument("--brightness", type=float, default=0,
                    help="Brightness adjustment value (0 to disable, typical: 30)")
parser.add_argument("--cursor-filter", type=str, default="all",
                    choices=["none", "kalman", "moving_average", "fir", "median", "exponential", "exp", "lowpass", "low_pass", "lowpass2", "low_pass2", "hampel", "all"],
                    help="Filter type for cursor smoothing (use 'all' to show all filters)")
parser.add_argument("--cursor", type=str, default=None,
                    choices=["none", "kalman", "moving_average", "fir", "median", "exponential", "exp", "lowpass", "low_pass", "lowpass2", "low_pass2", "hampel"],
                    help="Control Xorg cursor with specified filter")
parser.add_argument("--cursor-still", action="store_true",
                    help="Only move cursor when head movement is detected")
parser.add_argument("--cursor-relative", action="store_true",
                    help="Use relative cursor control (hold 'w' key to activate)")
parser.add_argument("--datasource", type=str, default="pitchyaw",
                    choices=["pitchyaw", "normalproj"],
                    help="Data source for cursor control: pitchyaw (Euler angles) or normalproj (face normal projection)")
parser.add_argument("--vector", type=str, default="location",
                    choices=["location", "speed"],
                    help="Vector interpretation mode: location (direct position) or speed (velocity)")
parser.add_argument("--inv", type=str, default="none",
                    choices=["none", "x", "y", "xy"],
                    help="Image inversion mode: none (default), x (mirror horizontally), y (flip vertically), xy (both)")
parser.add_argument("--gui", type=str, default="all",
                    choices=["all", "pointers", "cam", "none"],
                    help="GUI display mode: all (both windows), pointers (cursor window only), cam (camera only), none")
args = parser.parse_args()


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))


def get_screen_resolution():
    """Get screen resolution using xrandr."""
    try:
        output = subprocess.check_output(['xrandr']).decode('utf-8')
        # Find primary display resolution
        match = re.search(r'(\d+)x(\d+)\+\d+\+\d+', output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
    except:
        pass
    # Default fallback
    return 1920, 1080


def set_mouse_position(x, y):
    """Set mouse position using xdotool."""
    try:
        subprocess.run(['xdotool', 'mousemove', str(x), str(y)], check=True)
    except:
        pass


def get_mouse_position():
    """Get current mouse position using xdotool."""
    try:
        result = subprocess.run(['xdotool', 'getmouselocation'], 
                              capture_output=True, text=True, check=True)
        # Parse output like "x:123 y:456 screen:0 window:12345"
        match = re.search(r'x:(\d+) y:(\d+)', result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None, None


def move_mouse_relative(dx, dy):
    """Move mouse relative to current position."""
    try:
        subprocess.run(['xdotool', 'mousemove_relative', '--', str(dx), str(dy)], check=True)
    except:
        pass


def speed_mode_cursor_updater(velocity_ref, lock, stop_event):
    """Update cursor position based on velocity at 30 FPS."""
    update_interval = 1.0 / 30.0  # 30 FPS
    last_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time()
        dt = current_time - last_time
        
        if dt >= update_interval:
            with lock:
                vx = velocity_ref[0]
                vy = velocity_ref[1]
            
            if vx != 0 or vy != 0:
                # Apply velocity as pixels per frame (at 30 FPS)
                dx = int(vx)
                dy = int(vy)
                if dx != 0 or dy != 0:
                    move_mouse_relative(dx, dy)
            
            last_time = current_time
        
        # Small sleep to prevent busy waiting
        time.sleep(0.001)


def is_windows_key_pressed():
    """Check if Windows/Super key is currently pressed."""
    try:
        # Use xdotool to get the state of modifier keys
        result = subprocess.run(['xdotool', 'getactivewindow', 'getwindowpid'], 
                              capture_output=True, text=True)
        # Check if Super_L or Super_R is pressed using xinput
        result = subprocess.run(['xinput', 'query-state', 'keyboard'], 
                              capture_output=True, text=True, stderr=subprocess.DEVNULL)
        if 'Super' in result.stdout:
            return True
    except:
        pass
    
    # Alternative method using xdotool keystate (requires xdotool 3.20160805.1 or newer)
    try:
        for key in ['Super_L', 'Super_R', 'Super', 'Meta_L', 'Meta_R']:
            result = subprocess.run(['xdotool', 'keystate', '--clearmodifiers', key],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True
    except:
        pass
    return False


def get_face_normal_from_rotation_matrix(rotation_matrix):
    """Extract the face normal vector from rotation matrix.
    
    The face normal is the Z-axis of the rotated coordinate system,
    which is the third column of the rotation matrix.
    
    Args:
        rotation_matrix: 3x3 rotation matrix from cv2.Rodrigues
        
    Returns:
        numpy array: 3D normal vector (x, y, z)
    """
    # The third column of rotation matrix represents the Z-axis (face normal)
    # in the world coordinate system
    normal = rotation_matrix[:, 2]
    return normal


def project_normal_to_xy(normal_vector):
    """Project the 3D normal vector to 2D coordinates.
    
    Args:
        normal_vector: 3D normal vector (x, y, z)
        
    Returns:
        Tuple (x, y) representing the projection
    """
    # Direct projection onto X-Y plane
    # Note: We use -x to match the yaw direction convention
    return -normal_vector[0], -normal_vector[1]


def map_normal_to_cursor(normal_x, normal_y, window_width=800, window_height=600):
    """Map projected normal vector to cursor position.
    
    Args:
        normal_x: X component of projected normal (-1 to 1)
        normal_y: Y component of projected normal (-1 to 1)
        window_width: Width of the cursor window
        window_height: Height of the cursor window
        
    Returns:
        Tuple (x, y) of cursor position in window coordinates
    """
    # Map from -0.5 to 0.5 range to window dimensions
    # Clamp to reasonable range
    normal_x = max(-0.5, min(0.5, normal_x))
    normal_y = max(-0.5, min(0.5, normal_y))
    
    # Linear mapping
    x = int((normal_x + 0.5) * window_width)
    y = int((normal_y + 0.5) * window_height)
    
    return x, y


def map_angles_to_cursor(pitch, yaw, window_width=800, window_height=600):
    """Map pitch and yaw angles to cursor position in window.
    
    Args:
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        window_width: Width of the cursor window
        window_height: Height of the cursor window
    
    Returns:
        Tuple (x, y) of cursor position in window coordinates
    """
    # Map from -10 to 10 degrees to 0 to window dimensions
    # Center is at (window_width/2, window_height/2)
    
    # Clamp angles to -10 to 10 range
    pitch = max(-10, min(10, pitch))
    yaw = max(-10, min(10, yaw))
    
    # Linear mapping: -10 degrees = 0, 0 degrees = center, 10 degrees = max
    # Note: Y is inverted because negative pitch means looking up
    x = int((yaw + 10) / 20 * window_width)
    y = int((-pitch + 10) / 20 * window_height)
    
    return x, y


def run():
    # Before estimation started, there are some startup works to do.

    # Initialize the video source from webcam or video file.
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    
    # Reduce webcam buffer to minimize latency (only for webcam, not video files)
    if args.video is None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Webcam source: {video_src} (buffer size set to 1 for low latency)")
    else:
        print(f"Video source: {video_src}")
    
    # Create cursor window
    cursor_window_width = 800
    cursor_window_height = 600
    cv2.namedWindow("Head Pose Cursor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Head Pose Cursor", cursor_window_width, cursor_window_height)
    
    # Initialize cursor position at center
    cursor_x = cursor_window_width // 2
    cursor_y = cursor_window_height // 2
    
    # Check if we need to control Xorg cursor
    xorg_cursor_control = False
    xorg_cursor_filter = None
    screen_width, screen_height = 1920, 1080
    
    if args.cursor:
        xorg_cursor_control = True
        xorg_cursor_filter = create_cursor_filter(args.cursor)
        screen_width, screen_height = get_screen_resolution()
        print(f"Controlling Xorg cursor with {args.cursor} filter")
        print(f"Screen resolution: {screen_width}x{screen_height}")
        
    # Initialize movement detector if needed
    movement_detector = None
    if args.cursor_still:
        movement_detector = MovementDetector(window_size=15, std_threshold=2.0, range_threshold=5.0)
        print("Movement detection enabled - cursor only moves when head movement detected")
        
    # Initialize relative cursor control
    origin_pitch = 0.0
    origin_yaw = 0.0
    origin_normal_x = 0.0
    origin_normal_y = 0.0
    last_mouse_x = 0
    last_mouse_y = 0
    w_key_pressed = False
    
    # Speed mode variables (using lists as mutable containers)
    cursor_velocity = [0.0, 0.0]  # [x, y] velocity
    speed_mode_lock = threading.Lock()
    speed_update_thread = None
    stop_speed_thread = threading.Event()
    
    if args.cursor_relative:
        print(f"Relative cursor mode: Hold 'w' key to control cursor (vector mode: {args.vector})")
        
    # Print GUI mode
    if args.gui == "none":
        print("Running in headless mode (no GUI windows). Press Ctrl+C to exit.")
    elif args.gui == "cam":
        print("Showing camera preview only")
    elif args.gui == "pointers":
        print("Showing cursor/pointers window only")
    
    # Create cursor filters
    if args.cursor_filter == "all":
        # Create all filters with their colors
        filters = {
            "none": {"filter": create_cursor_filter("none"), "color": (255, 255, 255), "pos": (cursor_x, cursor_y)},  # White
            "kalman": {"filter": create_cursor_filter("kalman"), "color": (0, 255, 0), "pos": (cursor_x, cursor_y)},  # Green
            "median": {"filter": create_cursor_filter("median"), "color": (255, 0, 0), "pos": (cursor_x, cursor_y)},  # Blue
            "moving_avg": {"filter": create_cursor_filter("moving_average"), "color": (0, 255, 255), "pos": (cursor_x, cursor_y)},  # Yellow
            "exponential": {"filter": create_cursor_filter("exponential"), "color": (255, 0, 255), "pos": (cursor_x, cursor_y)},  # Magenta
            "lowpass": {"filter": create_cursor_filter("lowpass"), "color": (0, 165, 255), "pos": (cursor_x, cursor_y)},  # Orange
            "lowpass2": {"filter": create_cursor_filter("lowpass2"), "color": (255, 100, 100), "pos": (cursor_x, cursor_y)},  # Light Blue
            "hampel": {"filter": create_cursor_filter("hampel"), "color": (100, 255, 100), "pos": (cursor_x, cursor_y)},  # Light Green
        }
        print("Showing all cursor filters for comparison")
    else:
        # Single filter mode
        cursor_filter = create_cursor_filter(args.cursor_filter)
        print(f"Using cursor filter: {args.cursor_filter}")

    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    # Now, let the frames flow.
    try:
        while True:

            # Read a frame.
            frame_got, frame = cap.read()
            if frame_got is False:
                break
            
            # If the frame comes from webcam, flip it so it looks like a mirror.
            # Note: Default webcam behavior is to mirror (horizontal flip)
            if video_src == 0 and args.inv == "none":
                frame = cv2.flip(frame, 1)  # Horizontal flip for mirror effect
            
            # Apply image inversion based on --inv argument
            if args.inv == "x":
                # Flip horizontally (mirror)
                frame = cv2.flip(frame, 1)
            elif args.inv == "y":
                # Flip vertically
                frame = cv2.flip(frame, 0)
            elif args.inv == "xy":
                # Flip both horizontally and vertically
                frame = cv2.flip(frame, -1)
            
            # Apply brightness adjustment if requested
            if args.brightness > 0:
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=args.brightness)

            # Step 1: Get faces from current frame.
            faces, _ = face_detector.detect(frame, 0.7)

            # Any valid face found?
            if len(faces) > 0:
                tm.start()

                # Step 2: Detect landmarks. Crop and feed the face area into the
                # mark detector. Note only the first face will be used for
                # demonstration.
                face = refine(faces, frame_width, frame_height, 0.15)[0]
                x1, y1, x2, y2 = face[:4].astype(int)
                patch = frame[y1:y2, x1:x2]

                # Run the mark detection.
                marks = mark_detector.detect([patch])[0].reshape([68, 2])

                # Convert the locations from local face area to the global image.
                marks *= (x2 - x1)
                marks[:, 0] += x1
                marks[:, 1] += y1

                # Step 3: Try pose estimation with 68 points.
                pose = pose_estimator.solve(marks)
                
                # Get Euler angles and print them
                if pose is not None:
                    try:
                        rotation_vector, translation_vector = pose
                        
                        # Get rotation matrix for both modes
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                        
                        # Initialize variables for cursor control
                        if args.datasource == "normalproj":
                            # Extract face normal and project to 2D
                            face_normal = get_face_normal_from_rotation_matrix(rotation_matrix)
                            normal_x, normal_y = project_normal_to_xy(face_normal)
                            raw_x, raw_y = map_normal_to_cursor(normal_x, normal_y, cursor_window_width, cursor_window_height)
                            print(f"normal_x: {normal_x:.3f}, normal_y: {normal_y:.3f}, normal_z: {face_normal[2]:.3f}")
                            
                            # For movement detector, convert normal projection to approximate angles
                            # This is a rough approximation for compatibility
                            pitch = np.degrees(np.arcsin(-normal_y))
                            yaw = np.degrees(np.arcsin(-normal_x))
                        else:
                            # Original pitch/yaw mode
                            pitch, yaw, roll = pose_estimator.get_euler_angles(rotation_vector)
                            print(f"pitch: {pitch:.2f}, yaw: {yaw:.2f}, roll: {roll:.2f}")
                            raw_x, raw_y = map_angles_to_cursor(pitch, yaw, cursor_window_width, cursor_window_height)
                        
                        if args.cursor_filter == "all":
                            # Update all filters
                            for name, filter_data in filters.items():
                                filter_data["pos"] = filter_data["filter"].filter(raw_x, raw_y)
                        else:
                            # Single filter mode
                            cursor_x, cursor_y = cursor_filter.filter(raw_x, raw_y)
                        
                        # Update movement detector if enabled
                        is_moving = True  # Default to moving if no detector
                        if movement_detector:
                            is_moving = movement_detector.update(pitch, yaw)
                            # Debug output (optional)
                            if is_moving:
                                print("MOVING", end=" ")
                        
                        # Check for 'w' key state in relative mode
                        if args.cursor_relative and xorg_cursor_control:
                            # Unfortunately cv2.waitKey only detects key press events, not hold state
                            # For now, we'll use a toggle approach with 'w' key
                            pass  # Key handling is done at the end of the loop
                        
                        # Control Xorg cursor if enabled
                        if xorg_cursor_control and xorg_cursor_filter:
                            if args.cursor_relative:
                                # Relative mode - only move when w is pressed
                                if w_key_pressed:
                                    # Use head movement from origin as offset
                                    if args.datasource == "normalproj":
                                        # Use normal vector deltas
                                        delta_x = normal_x - origin_normal_x
                                        delta_y = normal_y - origin_normal_y
                                        # Scale factor for normal projection (adjustable)
                                        pixel_per_unit = 2000
                                        dx = int(delta_x * pixel_per_unit)
                                        dy = int(delta_y * pixel_per_unit)
                                    else:
                                        # Use pitch/yaw deltas
                                        delta_pitch = pitch - origin_pitch
                                        delta_yaw = yaw - origin_yaw
                                        # Map angle deltas to pixel movements (scale factor)
                                        # 1 degree = 50 pixels (adjustable)
                                        pixel_per_degree = 50
                                        dx = int(delta_yaw * pixel_per_degree)
                                        dy = int(-delta_pitch * pixel_per_degree)  # Inverted
                                    
                                    # Apply filter to the deltas
                                    filtered_dx, filtered_dy = xorg_cursor_filter.filter(dx, dy)
                                    
                                    if args.vector == "speed":
                                        # Speed mode: interpret deltas as velocity
                                        # Scale down for velocity (pixels per frame at 30 FPS)
                                        velocity_scale = 0.1  # Adjustable scale factor
                                        with speed_mode_lock:
                                            if is_moving:
                                                cursor_velocity[0] = filtered_dx * velocity_scale
                                                cursor_velocity[1] = filtered_dy * velocity_scale
                                            else:
                                                cursor_velocity[0] = 0.0
                                                cursor_velocity[1] = 0.0
                                    else:
                                        # Location mode (original behavior)
                                        if is_moving:
                                            move_mouse_relative(filtered_dx - last_mouse_x, filtered_dy - last_mouse_y)
                                            last_mouse_x = filtered_dx
                                            last_mouse_y = filtered_dy
                                else:
                                    # When key not pressed, stop velocity in speed mode
                                    if args.vector == "speed":
                                        with speed_mode_lock:
                                            cursor_velocity[0] = 0.0
                                            cursor_velocity[1] = 0.0
                            else:
                                # Absolute mode (original behavior)
                                # Get filtered position from the Xorg filter
                                xorg_x, xorg_y = xorg_cursor_filter.filter(raw_x, raw_y)
                                
                                # Only move cursor if movement detected (or detector disabled)
                                if is_moving:
                                    # Scale from window coordinates to screen coordinates
                                    screen_x = int(xorg_x * screen_width / cursor_window_width)
                                    screen_y = int(xorg_y * screen_height / cursor_window_height)
                                    # Set mouse position
                                    set_mouse_position(screen_x, screen_y)
                    except (ValueError, TypeError) as e:
                        print(f"Error extracting pose data: {e}")
                
                tm.stop()
            
            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            if 'pose' in locals() and pose is not None:
                pose_estimator.visualize(frame, pose, color=(0, 255, 0))
                
                # Always draw the normal vector
                pose_estimator.draw_normal_vector(frame, pose, color=(255, 255, 0))

            # Do you want to see the axes?
            # pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # if 'marks' in locals():
            #     mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # if 'faces' in locals():
            #     face_detector.visualize(frame, faces)

            # Draw the FPS on screen.
            cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Create cursor display image only if needed
            if args.gui in ["all", "pointers"]:
                cursor_img = np.zeros((cursor_window_height, cursor_window_width, 3), dtype=np.uint8)
                cursor_img.fill(50)  # Dark gray background
                
                # Draw center crosshair
                center_x = cursor_window_width // 2
                center_y = cursor_window_height // 2
                cv2.line(cursor_img, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
                cv2.line(cursor_img, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)
            
            # Draw cursor(s) and UI elements only if cursor window is shown
            if args.gui in ["all", "pointers"]:
                if args.cursor_filter == "all":
                    # Draw all cursors with their colors
                    for name, filter_data in filters.items():
                        x, y = filter_data["pos"]
                        color = filter_data["color"]
                        # Semi-transparent fill for overlapping cursors
                        overlay = cursor_img.copy()
                        cv2.circle(overlay, (x, y), 12, color, -1)
                        cv2.addWeighted(overlay, 0.6, cursor_img, 0.4, 0, cursor_img)
                        # Solid border
                        cv2.circle(cursor_img, (x, y), 12, color, 2)
                else:
                    # Single cursor
                    cv2.circle(cursor_img, (cursor_x, cursor_y), 15, (0, 255, 0), -1)
                    cv2.circle(cursor_img, (cursor_x, cursor_y), 15, (0, 150, 0), 2)
            
            # Add angle or normal vector text
            if args.datasource == "normalproj" and 'normal_x' in locals():
                cv2.putText(cursor_img, f"Normal X: {normal_x:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(cursor_img, f"Normal Y: {normal_y:.3f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(cursor_img, f"Normal Z: {face_normal[2]:.3f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            elif 'pitch' in locals() and 'yaw' in locals():
                cv2.putText(cursor_img, f"Pitch: {pitch:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(cursor_img, f"Yaw: {yaw:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Show movement status if detector is active
                if movement_detector and 'is_moving' in locals():
                    status_color = (0, 255, 0) if is_moving else (0, 100, 255)  # Green if moving, red if still
                    status_text = "MOVING" if is_moving else "STILL"
                    status_y = 120 if args.datasource == "normalproj" else 90
                    cv2.putText(cursor_img, f"Status: {status_text}", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add legend
            if args.cursor_filter == "all":
                # Draw legend at bottom
                legend_y = cursor_window_height - 30
                legend_x = 10
                for i, (name, filter_data) in enumerate(filters.items()):
                    color = filter_data["color"]
                    text_x = legend_x + i * 140
                    cv2.putText(cursor_img, name, (text_x, legend_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                filter_y = 120 if args.datasource == "normalproj" else 90
                cv2.putText(cursor_img, f"Filter: {args.cursor_filter}", (10, filter_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Add datasource info
            datasource_y = cursor_window_height - 60
            cv2.putText(cursor_img, f"Datasource: {args.datasource}", (10, datasource_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Add vector mode info if in relative mode
            if args.cursor_relative:
                vector_y = cursor_window_height - 40
                cv2.putText(cursor_img, f"Vector mode: {args.vector}", (10, vector_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Show relative mode status if active
            if args.cursor_relative and xorg_cursor_control:
                mode_color = (0, 255, 0) if w_key_pressed else (200, 200, 200)
                mode_text = "'w' key HELD" if w_key_pressed else "Hold 'w' key"
                rel_mode_y = 150 if args.datasource == "normalproj" else 120
                cv2.putText(cursor_img, f"Relative mode: {mode_text}", (10, rel_mode_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
                if w_key_pressed:
                    if args.datasource == "normalproj":
                        cv2.putText(cursor_img, f"Origin: X={origin_normal_x:.3f} Y={origin_normal_y:.3f}", (10, 145),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    else:
                        cv2.putText(cursor_img, f"Origin: P={origin_pitch:.1f} Y={origin_yaw:.1f}", (10, 145),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                        
                    # Show velocity in speed mode
                    if args.vector == "speed":
                        with speed_mode_lock:
                            vel_x = cursor_velocity[0]
                            vel_y = cursor_velocity[1]
                        cv2.putText(cursor_img, f"Velocity: X={vel_x:.1f} Y={vel_y:.1f} px/frame", (10, 170),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Show windows based on --gui setting
            if args.gui in ["all", "cam"]:
                cv2.imshow("Preview", frame)
            if args.gui in ["all", "pointers"]:
                cv2.imshow("Head Pose Cursor", cursor_img)
        
            # Handle key presses
            if args.gui != "none":
                key = cv2.waitKey(1)
            else:
                # In headless mode, we need a small delay but no key handling
                key = -1
                time.sleep(0.001)  # Small delay to prevent CPU spinning
            
            if key == 27 and args.gui != "none":  # ESC to exit (only works with GUI)
                # Clean up speed thread if running
                if speed_update_thread is not None:
                    stop_speed_thread.set()
                    speed_update_thread.join(timeout=1.0)
                break
            elif key == ord('w') and args.cursor_relative and xorg_cursor_control:
                # Toggle w key state and capture origin when pressed
                w_key_pressed = not w_key_pressed
                if w_key_pressed:
                    if args.datasource == "normalproj" and 'normal_x' in locals():
                        origin_normal_x = normal_x
                        origin_normal_y = normal_y
                        print(f"'w' key pressed. Origin: normal_x={origin_normal_x:.3f}, normal_y={origin_normal_y:.3f}")
                    elif 'pitch' in locals() and 'yaw' in locals():
                        origin_pitch = pitch
                        origin_yaw = yaw
                        print(f"'w' key pressed. Origin: pitch={origin_pitch:.1f}, yaw={origin_yaw:.1f}")
                    last_mouse_x = 0
                    last_mouse_y = 0
                    if xorg_cursor_filter:
                        xorg_cursor_filter.reset()
                
                    # Start speed thread if in speed mode
                    if args.vector == "speed" and speed_update_thread is None:
                        stop_speed_thread.clear()
                        speed_update_thread = threading.Thread(
                            target=speed_mode_cursor_updater,
                            args=(cursor_velocity, speed_mode_lock, stop_speed_thread)
                        )
                        speed_update_thread.daemon = True
                        speed_update_thread.start()
                        print("Speed mode thread started")
                else:
                    print("'w' key released")
                    # Stop speed thread if in speed mode
                    if args.vector == "speed" and speed_update_thread is not None:
                        stop_speed_thread.set()
                        speed_update_thread.join(timeout=1.0)
                        speed_update_thread = None
                        with speed_mode_lock:
                            cursor_velocity[0] = 0.0
                            cursor_velocity[1] = 0.0
                            print("Speed mode thread stopped")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Clean up speed thread if running
        if speed_update_thread is not None:
            stop_speed_thread.set()
            speed_update_thread.join(timeout=1.0)
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
