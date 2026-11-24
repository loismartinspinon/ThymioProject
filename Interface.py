"""
Objective: display the Thymio on the visual interface of the camera with its rectified path, the Thymio's position, its angular position and the arrival position.

Implementation:
- OpenCV: used to fetch the image content, to display the camera visual into a separated window and to fetch the position of the green, red and blue points
- Trajectory computation: one decided to implement a A* method. This can be done as one knows the initial and final position.

Use case:
- Only called in the file main.ipynb in order to display the interface from the calibration until the Thymio reaches the final point.

User commands:
- q to quit
- c to recalibrate the camera
"""

# =============================================================================
# Import of libraries and Vision methods and variables
# =============================================================================

import time
import numpy as np
import cv2

from Vision import MAP_WIDTH, MAP_HEIGHT, rectify_image, fetch_status_cv


# =============================================================================
# Parameters
# =============================================================================

CAMERA_INDEX = 0           # index of the camera given to cv2.VideoCapture
CAMERA_WIDTH = 1280        # camera resolution (width) [???]
CAMERA_HEIGHT = 720        # camera resolution (height) [???]

SNAPSHOT_PERIOD = 0.2     # time difference between 2 updates [s]
CALIBRATION_FRAMES = 10    # number of frames with 4 QR codes in order to confirm the calibration
MAP_SCALE = 4              # display zoom of the rectified map


HEADING_LENGTH = 15        # arrow length for the Thymio's direction
HEADING_OFFSET = -np.pi / 2 # arrow angular position for the Thymio's direction, counterclockwise is considered as negative and clockwise is considered as positive, where 0° is the angular orientation of the displayed direction vector

WINDOW_CAMERA = "Thymio - Camera Live" # window name for the camera visual
WINDOW_MAP = "Thymio - Map"            # window name for the camera visual with planned path

# Corners of the rectified map, in map pixels (TL, TR, BR, BL)

# Corners positions, defined as in map pixels (TL, TR, BR, BL). TL = top left, TR = top right
MAP_CORNERS = np.array([
    [0, 0],
    [MAP_WIDTH - 1, 0],
    [MAP_WIDTH - 1, MAP_HEIGHT - 1],
    [0, MAP_HEIGHT - 1]
], dtype=np.float32)


# =============================================================================
# Camera calibration
# Calibration is only done once at the beginning of the entire program or if the user presses the button "q". The live camera waits to have the 4 QR codes recognized and then the windows disappears. From the defined QR codes positions, one rectifies the visual in order to have a rectangular image using the rectify_image() in the Vision.py file. The rectification process defines the homography H from the 4 map corners and is computed for every frame. Hence, H displays the rectified map and H_inv draws on the live camera.
# =============================================================================



# Apply the homography H to a list of points (x,y)
def project_points(points, H):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# Calibration from given frame: returns the homographic matrice H and its inverse
def calibration_from_image(image):
    _, _, H = rectify_image(image)
    return H, np.linalg.inv(H)


# Calibration from an entire video: returns the homographic matrice H and its inverse
def calibrate_camera(cap):
    # list of corners
    corners_list = []

    # while loop that will define the rectangular lines that are connected by the 4 QR codes, the process is repeated CALIBRATION_FRAMES times.
    while len(corners_list) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        # case where the frame cannot be fetched from the video
        if not ret:
            raise RuntimeError("Unable to read a frame from the camera")

        display = frame.copy()
        try:
            # fetches the position of the markers and the homographic matrice H
            _, frame_markers, H = rectify_image(frame)
            # adds the markers positions to the corners_list array
            corners_list.append(project_points(MAP_CORNERS, np.linalg.inv(H)))
            # considers all the marker positions and connect them through lines
            for qr in frame_markers:
                cv2.polylines(display, [qr.astype(np.int32)], True, (0, 255, 0), 2)
            # text displayed on interface about amount of marker positions sensed
            status, color = f"Calibration: {len(corners_list)}/{CALIBRATION_FRAMES}", (0, 255, 0)
        except ValueError as e:
            # text displayed in case no QR code is not 
            status, color = f"Calibration: {e}", (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "q: quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW_CAMERA, display)

        # if loop to close the camera visual window if "q" pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    # Median of the map corners seen by the camera, allows to get the final homography for the entire program. Returns the homography matrice H and its index.
    corners = np.median(np.array(corners_list), axis=0).astype(np.float32)
    H = cv2.getPerspectiveTransform(corners, MAP_CORNERS)
    return H, np.linalg.inv(H)


    
# =============================================================================
# Fetches all the important points for each frame every SNAPSHOT_PERIODS seconds.
# fetch_status_cv() looks for 4 QR codes for every snapshot. Hence, they need to be visible on the camera.
# Elements measures:
#    Thymio's position: (x,y) in map pixels. Defined as None if fetch_status_cv() failed.
#    theta: angle of the Thymio in radians (MORE DESCRIPTION NEEDED)
#    path: defined path with A* method, defined as a list of (x,y) points, None if no path found
#    next: next position to be aimed, considered as the 1st element of the path array
#    error: ValueError message given by fetch_status_cv()
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
# Camera window: live visual camera with calibrated area, the Thymio point and heading, the final point and the planned path projected with H_inv
# Map window: rectified vision of the last snapshot
# =============================================================================


# position coordinate conversion into 2 integer coordinates
def to_int(p):
    return (int(round(p[0])), int(round(p[1])))


# heading point coordinates definition, defined as the arrow starting from the point between the red and green points and is perpendicular to the red-green segment. returns the final position of the heading array
def heading_point(state):
    x, y = state["thymio"]
    heading = state["theta"] + HEADING_OFFSET
    return (x + HEADING_LENGTH * np.cos(heading),
            y + HEADING_LENGTH * np.sin(heading))


# defines the status of the planned path. Display the position, theta angle and next position if the frame is available and calibrated. However, in other cases, no status information is given.
def status_lines(state, stale):
    # Text lines (text, color) shown on the camera window
    lines = []
    # case of Thymio's position defined
    if state["thymio"] is not None:
        x, y = state["thymio"]
        lines.append((f"Position: ({x:.1f}, {y:.1f}) px", (0, 255, 0)))
        lines.append((f"Theta: {np.degrees(state['theta']):.1f} deg", (0, 255, 0)))
        lines.append((f"Next position: {state['next']}", (0, 255, 255)))
    # TO BE DEFINED
    if stale:
        lines.append(("Last known position (not updated)", (0, 0, 255)))
    # case where status is defined as "error"
    if state["error"] is not None:
        lines.append((state["error"], (0, 0, 255)))
    # case where no planned path or Thymio's position is defined
    elif state["path"] is None and state["thymio"] is not None:
        lines.append(("No path found", (0, 0, 255)))
    return lines

# TO BE EXPLAINED
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

# TO BE EXPLAINED
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
# Stepd followed:
#    1. Camera turned on and calibration with 4 QR codes
#    2. Every SNAPSHOT_PERIOD seconds, status is fetched with fetch_status_cv()
#    3. Live camera displayed  with last know position
# =============================================================================


def run_interface():
    # Camera turned on, configurations setup
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Case of closed camera
    if not cap.isOpened():
        print("Error: unable to open the camera")
    else:
        try:
            # Real resolution printed
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera activated! Resolution: {width} x {height}")
            print("Calibration... (show the 4 QR codes, q to quit)")
            calibration = calibrate_camera(cap)

            # Calibration status check
            if calibration is None:
                print("Calibration aborted.")
            else:
                # Homographic matrice H and its inverse defined
                H, H_inv = calibration
                print("Calibration done. q: quit, c: recalibrate")

                state = None          # result of the last snapshot, initially defined as None
                last_pose = None      # last detected (x, y, theta), initially defined as None
                last_snapshot = 0.0

                # Starts analysing every frame of th camera until final position is reached
                while True:
                    ret, frame = cap.read()
                    # Frame presence check
                    if not ret:
                        print("Unable to read a frame from the camera")
                        break

                    # Snapshot every SNAPSHOT_PERIOD seconds
                    now = time.time()
                    # If condition for ensuring that the loop doesn't start before the previous loop time + SNAPSHOT_PERIOD
                    if now - last_snapshot >= SNAPSHOT_PERIOD:
                        last_snapshot = now
                        state = take_snapshot(frame)
                        stale = state["thymio"] is None
                        # TO BE DEFINED
                        if stale and last_pose is not None:
                            # Keep displaying the last known position
                            state["thymio"], state["theta"] = last_pose[:2], last_pose[2]
                        elif not stale:
                            last_pose = (*state["thymio"], state["theta"])
                        rectified = cv2.warpPerspective(frame, H, (MAP_WIDTH, MAP_HEIGHT))
                        cv2.imshow(WINDOW_MAP, draw_map(rectified, state, stale))

                    cv2.imshow(WINDOW_CAMERA, draw_camera(frame, state, H_inv, state is not None and stale))

                    # Keyboard user command
                    key = cv2.waitKey(1) & 0xFF
                    # If "q" pressed, leaves the run_interface() method
                    if key == ord('q'):
                        break
                    # If "c" pressed, starts again the calibration of the image
                    if key == ord('c'):
                        calibration = calibrate_camera(cap)
                        if calibration is None:
                            break
                        H, H_inv = calibration
                        print("Recalibration done.")
                        
        # Close the visual inteface window
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera closed.")

# TO BE DESCRIBED
if __name__ == "__main__":
    run_interface()
