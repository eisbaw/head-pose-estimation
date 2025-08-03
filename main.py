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

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
from cursor_filters import create_cursor_filter, CursorFilter

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
parser.add_argument("--brightness", type=float, default=0,
                    help="Brightness adjustment value (0 to disable, typical: 30)")
parser.add_argument("--cursor-filter", type=str, default="all",
                    choices=["none", "kalman", "moving_average", "fir", "median", "exponential", "exp", "lowpass", "low_pass", "all"],
                    help="Filter type for cursor smoothing (use 'all' to show all filters)")
args = parser.parse_args()


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))


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
    # Map from -20 to 20 degrees to 0 to window dimensions
    # Center is at (window_width/2, window_height/2)
    
    # Clamp angles to -20 to 20 range
    pitch = max(-20, min(20, pitch))
    yaw = max(-20, min(20, yaw))
    
    # Linear mapping: -20 degrees = 0, 0 degrees = center, 20 degrees = max
    # Note: Y is inverted because negative pitch means looking up
    x = int((yaw + 20) / 40 * window_width)
    y = int((-pitch + 20) / 40 * window_height)
    
    return x, y


def run():
    # Before estimation started, there are some startup works to do.

    # Initialize the video source from webcam or video file.
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    print(f"Video source: {video_src}")
    
    # Create cursor window
    cursor_window_width = 800
    cursor_window_height = 600
    cv2.namedWindow("Head Pose Cursor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Head Pose Cursor", cursor_window_width, cursor_window_height)
    
    # Initialize cursor position at center
    cursor_x = cursor_window_width // 2
    cursor_y = cursor_window_height // 2
    
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
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)
        
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
                    pitch, yaw, roll = pose_estimator.get_euler_angles(rotation_vector)
                    print(f"pitch: {pitch:.2f}, yaw: {yaw:.2f}, roll: {roll:.2f}")
                    
                    # Update cursor position based on pitch and yaw
                    raw_x, raw_y = map_angles_to_cursor(pitch, yaw, cursor_window_width, cursor_window_height)
                    
                    if args.cursor_filter == "all":
                        # Update all filters
                        for name, filter_data in filters.items():
                            filter_data["pos"] = filter_data["filter"].filter(raw_x, raw_y)
                    else:
                        # Single filter mode
                        cursor_x, cursor_y = cursor_filter.filter(raw_x, raw_y)
                except (ValueError, TypeError) as e:
                    print(f"Error extracting pose data: {e}")
                    continue

            tm.stop()

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            # pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # face_detector.visualize(frame, faces)

        # Draw the FPS on screen.
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Create cursor display image
        cursor_img = np.zeros((cursor_window_height, cursor_window_width, 3), dtype=np.uint8)
        cursor_img.fill(50)  # Dark gray background
        
        # Draw center crosshair
        center_x = cursor_window_width // 2
        center_y = cursor_window_height // 2
        cv2.line(cursor_img, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
        cv2.line(cursor_img, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)
        
        # Draw cursor(s)
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
        
        # Add angle text
        if 'pitch' in locals() and 'yaw' in locals():
            cv2.putText(cursor_img, f"Pitch: {pitch:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(cursor_img, f"Yaw: {yaw:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
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
            cv2.putText(cursor_img, f"Filter: {args.cursor_filter}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Show windows
        cv2.imshow("Preview", frame)
        cv2.imshow("Head Pose Cursor", cursor_img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    run()
