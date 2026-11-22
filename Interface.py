"""
Interface: live display of the Thymio on the camera image and on the rectified map.

- Actual position and orientation of the Thymio
- Projected A* trajectory to the goal

The position, angle and path are computed by fetch_status_cv (Vision_test.py);
this file only calibrates the camera for the display and draws the result.

Usage:
    python Interface.py              -> live interface with the camera
    from Interface import run_interface, take_snapshot, ...   (e.g. in main.py)

Keys in the camera window: q to quit, c to recalibrate the camera.
"""
import time
import numpy as np
import cv2

from Vision import MAP_WIDTH, MAP_HEIGHT, rectify_image, fetch_status_cv


# =============================================================================
# Parameters
# =============================================================================

CAMERA_INDEX = 0           # index of the camera given to cv2.VideoCapture
CAMERA_WIDTH = 1280        # requested camera resolution (DSHOW default is often 640 x 480,
CAMERA_HEIGHT = 720        # too small to detect the QR codes reliably)
SNAPSHOT_PERIOD = 0.2     # seconds between two position updates
CALIBRATION_FRAMES = 10    # frames with the 4 QR codes needed to calibrate
MAP_SCALE = 4              # display zoom of the rectified map (205 x 155 px is small)
HEADING_LENGTH = 15        # length of the heading arrow, in map pixels
# Direction of the heading arrow relative to theta (red -> green vector of fetch_status_cv):
# -pi/2 = rotated 90 deg counter-clockwise on screen (y axis points down),
# i.e. perpendicular to the red-green segment, starting from its middle
HEADING_OFFSET = -np.pi / 2

WINDOW_CAMERA = "Thymio - Camera Live"
WINDOW_MAP = "Thymio - Map"

# Corners of the rectified map, in map pixels (TL, TR, BR, BL)
MAP_CORNERS = np.array([
    [0, 0],
    [MAP_WIDTH - 1, 0],
    [MAP_WIDTH - 1, MAP_HEIGHT - 1],
    [0, MAP_HEIGHT - 1]
], dtype=np.float32)


# =============================================================================
# Camera calibration
# Done once at the start (and again with the 'c' key if the camera moves).
# The live camera is displayed until the 4 QR codes have been detected on CALIBRATION_FRAMES
# frames. For each frame, rectify_image (Vision_test.py) gives the homography H (camera -> map);
# the 4 map corners are projected back into the camera image and the median over the frames is
# kept, which makes the calibration robust to a bad detection on one frame.
# Only used for the display: H shows the rectified map, H_inv draws on the live camera.
# =============================================================================

def project_points(points, M):
    # Apply the homography M to a list of (x, y) points
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def calibration_from_image(image):
    # Calibration from a single image (raises ValueError if the 4 QR codes are not found)
    _, _, H = rectify_image(image)
    return H, np.linalg.inv(H)


def calibrate_camera(cap):
    # Live calibration. Returns (H, H_inv), or None if 'q' is pressed.
    corners_list = []

    while len(corners_list) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to read a frame from the camera")

        display = frame.copy()
        try:
            _, frame_markers, H = rectify_image(frame)
            corners_list.append(project_points(MAP_CORNERS, np.linalg.inv(H)))
            for qr in frame_markers:
                cv2.polylines(display, [qr.astype(np.int32)], True, (0, 255, 0), 2)
            status, color = f"Calibration: {len(corners_list)}/{CALIBRATION_FRAMES}", (0, 255, 0)
        except ValueError as e:
            status, color = f"Calibration: {e}", (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "q: quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW_CAMERA, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    # Median of the map corners seen by the camera -> final homography
    corners = np.median(np.array(corners_list), axis=0).astype(np.float32)
    H = cv2.getPerspectiveTransform(corners, MAP_CORNERS)
    return H, np.linalg.inv(H)


# =============================================================================
# Thymio localisation
# Every SNAPSHOT_PERIOD seconds, the snapshot is given to fetch_status_cv (Vision_test.py).
# Returns a dict:
#   thymio: (x, y) in map pixels, None if fetch_status_cv failed
#   theta:  angle in radians (red -> green vector)
#   path:   A* path as (x, y) points, None if no path
#   next:   next position to aim at (path[1]), None if no path
#   error:  message of the ValueError raised by fetch_status_cv, else None
# fetch_status_cv searches the 4 QR codes on every snapshot, so they must stay visible.
# =============================================================================

def take_snapshot(frame):
    try:
        x, y, theta, path = fetch_status_cv(frame)
    except ValueError as e:
        return {"thymio": None, "theta": None, "path": None, "next": None, "error": str(e)}

    next_position = None
    if path is not None:
        path = [(col, row) for row, col in path]    # (row, col) cells -> (x, y) points
        next_position = path[1] if len(path) > 1 else path[0]

    return {"thymio": (x, y), "theta": theta, "path": path, "next": next_position, "error": None}


# =============================================================================
# Data display
# Camera window: live camera with the calibrated area, the Thymio (point + heading) and the A*
# path, projected from the map into the camera image with H_inv.
# Map window: rectified top view of the last snapshot with the same information.
# =============================================================================

def to_int(p):
    return (int(round(p[0])), int(round(p[1])))


def heading_point(state):
    # End of the heading arrow: starts at the middle of the red and green points,
    # perpendicular to the red-green segment (theta rotated by HEADING_OFFSET)
    x, y = state["thymio"]
    heading = state["theta"] + HEADING_OFFSET
    return (x + HEADING_LENGTH * np.cos(heading),
            y + HEADING_LENGTH * np.sin(heading))


def status_lines(state, stale):
    # Text lines (text, color) shown on the camera window
    lines = []
    if state["thymio"] is not None:
        x, y = state["thymio"]
        lines.append((f"Position: ({x:.1f}, {y:.1f}) px", (0, 255, 0)))
        lines.append((f"Theta: {np.degrees(state['theta']):.1f} deg", (0, 255, 0)))
        lines.append((f"Next position: {state['next']}", (0, 255, 255)))
    if stale:
        lines.append(("Last known position (not updated)", (0, 0, 255)))
    if state["error"] is not None:
        lines.append((state["error"], (0, 0, 255)))
    elif state["path"] is None and state["thymio"] is not None:
        lines.append(("No path found", (0, 0, 255)))
    return lines


def draw_camera(frame, state, H_inv, stale):
    # Calibrated area
    area = project_points(MAP_CORNERS, H_inv).astype(np.int32)
    cv2.polylines(frame, [area], True, (255, 255, 0), 2)

    if state is None:
        return frame

    if state["thymio"] is not None:
        thymio, head = project_points([state["thymio"], heading_point(state)], H_inv)
        color = (0, 0, 255) if not stale else (128, 128, 128)
        cv2.arrowedLine(frame, to_int(thymio), to_int(head), color, 2, tipLength=0.3)
        cv2.circle(frame, to_int(thymio), 8, color, -1)

    if state["path"] is not None:
        path = project_points(state["path"], H_inv).astype(np.int32)
        cv2.polylines(frame, [path], False, (0, 255, 255), 2)

    for i, (text, color) in enumerate(status_lines(state, stale)):
        cv2.putText(frame, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def draw_map(rectified, state, stale):
    s = MAP_SCALE
    img = cv2.resize(rectified, (MAP_WIDTH * s, MAP_HEIGHT * s), interpolation=cv2.INTER_NEAREST)

    def m(p):
        return (int(round(p[0] * s)), int(round(p[1] * s)))

    if state["thymio"] is not None:
        color = (0, 0, 255) if not stale else (128, 128, 128)
        cv2.arrowedLine(img, m(state["thymio"]), m(heading_point(state)), color, 2, tipLength=0.3)
        cv2.circle(img, m(state["thymio"]), 8, color, -1)

    if state["path"] is not None:
        pts = np.array([m(p) for p in state["path"]], dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2)

    return img


# =============================================================================
# Test on a static image (without camera)
# =============================================================================

def test_static_image(image_path="ThymioSetup4.jfif"):
    # Calibration + fetch_status_cv on an image, to check the pipeline without the camera
    image = cv2.imread(image_path)
    H, H_inv = calibration_from_image(image)
    state = take_snapshot(image)
    print("Thymio:", state["thymio"], "theta:", state["theta"], "next:", state["next"], "error:", state["error"])
    print("Path length:", None if state["path"] is None else len(state["path"]))

    rectified = cv2.warpPerspective(image, H, (MAP_WIDTH, MAP_HEIGHT))
    cv2.imshow(WINDOW_CAMERA, draw_camera(image.copy(), state, H_inv, stale=False))
    cv2.imshow(WINDOW_MAP, draw_map(rectified, state, stale=False))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# Live interface
# 1. The camera turns on and calibrates itself (show the 4 QR codes)
# 2. Every SNAPSHOT_PERIOD seconds, a snapshot is given to fetch_status_cv
# 3. Between two snapshots, the live camera is displayed with the last known position
# =============================================================================

def run_interface():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    if not cap.isOpened():
        print("Error: unable to open the camera")
    else:
        try:
            # The camera may not support the requested resolution: print the real one
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera activated! Resolution: {width} x {height}")
            print("Calibration... (show the 4 QR codes, q to quit)")
            calibration = calibrate_camera(cap)

            if calibration is None:
                print("Calibration aborted.")
            else:
                H, H_inv = calibration
                print("Calibration done. q: quit, c: recalibrate")

                state = None          # result of the last snapshot
                last_pose = None      # last detected (x, y, theta)
                last_snapshot = 0.0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Unable to read a frame from the camera")
                        break

                    # Snapshot every SNAPSHOT_PERIOD seconds
                    now = time.time()
                    if now - last_snapshot >= SNAPSHOT_PERIOD:
                        last_snapshot = now
                        state = take_snapshot(frame)
                        stale = state["thymio"] is None
                        if stale and last_pose is not None:
                            # Keep displaying the last known position
                            state["thymio"], state["theta"] = last_pose[:2], last_pose[2]
                        elif not stale:
                            last_pose = (*state["thymio"], state["theta"])
                        rectified = cv2.warpPerspective(frame, H, (MAP_WIDTH, MAP_HEIGHT))
                        cv2.imshow(WINDOW_MAP, draw_map(rectified, state, stale))

                    cv2.imshow(WINDOW_CAMERA, draw_camera(frame, state, H_inv, state is not None and stale))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('c'):
                        calibration = calibrate_camera(cap)
                        if calibration is None:
                            break
                        H, H_inv = calibration
                        print("Recalibration done.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera closed.")


if __name__ == "__main__":
    run_interface()
