"""
Objective: extract the data from the video in order to fetch the Thymio's position and angle, final position and the position of obstacles

Implementation:
- OpenCV: used to fetch the position of the green, red and blue points, and to define the visibility graph with obstacles
- Trajectory computation: one decided to implement a A* method. This can be done as one knows the initial and final position.

Use case:
- File interface.py in order to display the defined status points into the interface

This is the only file not used on the main.ipynb because it is indirectly used in the interface.py file.
"""

# =============================================================================
# Import of libraries
# =============================================================================

import cv2
import numpy as np
import heapq


# =============================================================================
# Parameters
# =============================================================================

# Size of the rectified map in pixels
# Arena 100 x 84.375 cm (3 x 3 A3 tiles, outer corners of the ArUco markers) -> 2.56 px/cm on both axes
# (previous arena: 100 x 56.25 cm, 16:9 -> 256 x 144)
MAP_WIDTH = 128
MAP_HEIGHT = 108

# HSV color ranges
LOWER_RED1, UPPER_RED1 = np.array([0, 100, 100]), np.array([10, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([160, 100, 100]), np.array([180, 255, 255])
LOWER_GREEN, UPPER_GREEN = np.array([35, 80, 80]), np.array([90, 255, 255])
LOWER_BLUE, UPPER_BLUE = np.array([100, 120, 30]), np.array([150, 255, 220])

# Obstacles
GREY_THRESHOLD = 60        # threshold defining an obstacle color. if below this limit, it's consider as an obstacle
MIN_OBSTACLE_AREA = 40     # blobs smaller than this are noise, used to avoid considering Thymio's wheels as obstacles due to their color

# Thymio's half-size in pixels: half-width 5.5 cm + 0.5 cm margin = 6 cm -> 6 * 2.56 = 15 px. Jonas : J'ai augmenté un peux (ancienne valeurs : 15)
ROBOT_RADIUS_PX = 10

nest_position = []

# ArUco: dictionnaire 4x4, IDs 0-3 (comme dans les PDF de l'arène)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # coins plus précis -> meilleure homographie
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# ID du marqueur dans chaque coin de l'arène
ID_TL, ID_TR, ID_BR, ID_BL = 3, 2, 1, 0

def detect_aruco_markers(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(grey)
    if ids is None:
        return {}
    # {id: tableau (4, 2) des coins du marqueur}, en ignorant les IDs inconnus
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten()) if int(i) in (0, 1, 2, 3)}


# Orders the corners by the following order: top left, top right, bottom right, bottom left
def order_corners(p):
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).flatten()
    return np.array([
        p[np.argmin(s)],  # Top left, defined with smallest sum
        p[np.argmin(d)],  # Top right, defined with smallest difference
        p[np.argmax(s)],  # Bottom right, defined with largest sum
        p[np.argmax(d)]   # Bottom left, defined with largest difference
    ], dtype=np.float32)


# 
def rectify_image(image):
    found = detect_aruco_markers(image)
    missing = [i for i in (0, 1, 2, 3) if i not in found]
    if missing:
        raise ValueError(f"ArUco markers missing: {missing}")   # garder ValueError !

    # Coin extérieur de chaque marqueur = son coin le plus éloigné du centre de l'arène
    centre = np.mean([found[i].mean(axis=0) for i in found], axis=0)
    def outer(i):
        c = found[i]
        return c[np.argmax(np.linalg.norm(c - centre, axis=1))]

    src = np.array([outer(ID_TL), outer(ID_TR), outer(ID_BR), outer(ID_BL)], dtype=np.float32)
    markers = np.array([found[i] for i in (ID_TL, ID_TR, ID_BR, ID_BL)], dtype=np.float32)  # (4, 4, 2)

    dst = np.array([
        [0, 0],
        [MAP_WIDTH - 1, 0],
        [MAP_WIDTH - 1, MAP_HEIGHT - 1],
        [0, MAP_HEIGHT - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    rectified_image = cv2.warpPerspective(image, H, (MAP_WIDTH, MAP_HEIGHT))

    return rectified_image, markers, H

# Defines the color points mapping used for red, green and blue points
def color_masks(rectified_image):
    # frame image converted into the color system
    hsv = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2HSV)
    
    #WHY IS THAT DONE LIKE THAT? DIFFERENT FROM OTHER ONES
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
        cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    )
    # point position map definition
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    return mask_red, mask_green, mask_blue


# Definition of the mean points of each mapped color points in order to get a precise coordinate point for the actual and final position
def find_position(mask):
    # Centroid definition for each point
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the contour of each color point in the map
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    if not valid_contours:
        return None

    # NEEDS TO BE DEFINED
    M = cv2.moments(max(valid_contours, key=cv2.contourArea))
    if M["m00"] == 0:
        return None

    # Defines the centroid of each point
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# Definition of the visiblity graph
def obstacle_map(rectified_image, color_mask, markers, H):
    # Image conversion into grey colors
    grey_image = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)

    # Dark pixels, except the colored points
    color_mask = cv2.dilate(color_mask, np.ones((3, 3), np.uint8), iterations=1)
    obstacle_mask = (grey_image <= GREY_THRESHOLD) & (color_mask == 0)

    # QR codes removed: corners projected with the homographic matrice H
    qr_mask = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
    qr_corners_rectified = cv2.perspectiveTransform(markers.reshape(-1, 1, 2), H).reshape(4, 4, 2)
    # QR removal for each marker
    for qr in qr_corners_rectified:
        cv2.fillConvexPoly(qr_mask, np.round(qr).astype(np.int32), 255)
    qr_mask = cv2.dilate(qr_mask, np.ones((5, 5), np.uint8), iterations=1)
    obstacle_mask &= (qr_mask == 0)

    # Small noise blobs removed
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        obstacle_mask.astype(np.uint8), connectivity=8
    )
    # For loop for each small noise considered
    for label in range(1, n_labels):  # label 0 = background
        if stats[label, cv2.CC_STAT_AREA] < MIN_OBSTACLE_AREA:
            obstacle_mask[labels == label] = False

    return obstacle_mask


# Obstacles positions for the display: list of (cx, cy, x, y, w, h) in map pixels (centre + bounding box)
def obstacle_positions(obstacle_mask):
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(obstacle_mask.astype(np.uint8), connectivity=8)
    return [(*centroids[k], *stats[k, :4]) for k in range(1, n_labels)]


# Visibility graph definition
def occupancy_grid_from(obstacle_mask):
    # Add a margin to the obstacles in order to consider the half-width of the Thymio
    kernel_size = 2 * ROBOT_RADIUS_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated = cv2.dilate(obstacle_mask.astype(np.uint8), kernel)

    # Fills the visibility graph with ones for each pixels included in the margin
    inflated[:ROBOT_RADIUS_PX, :] = 1
    inflated[-ROBOT_RADIUS_PX:, :] = 1
    inflated[:, :ROBOT_RADIUS_PX] = 1
    inflated[:, -ROBOT_RADIUS_PX:] = 1

    return inflated.astype(bool)   # True means forbidden cell


# Heuristic function defined with octile distance
# DO WE REALLY WANT TO USE THIS NORM?
def heuristic(a, b):
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)


# A* planning path method
def a_star(grid, start, goal):
    # Defines the number of rows and columns
    rows, cols = grid.shape

    # 8 possible moves used in this A* method
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
             (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

    
    open_set = [(heuristic(start, goal), 0.0, start)]
    
    # List of previous positions
    came_from = {}
    
    # Cost
    g_score = {start: 0.0}
    closed = set()

    # Since open_set is defined, repeats this process
    # PRECISE DESCRIPTION OF A* METHOD
    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in closed:
            continue
        closed.add(current)

        for d_row, d_col, cost in moves:
            neighbor = (current[0] + d_row, current[1] + d_col)

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor] or neighbor in closed:
                continue

            # No corner cutting: a diagonal move needs both side cells free
            if d_row != 0 and d_col != 0:
                if grid[current[0] + d_row, current[1]] or grid[current[0], current[1] + d_col]:
                    continue

            new_g = g + cost
            if new_g < g_score.get(neighbor, np.inf):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_g + heuristic(neighbor, goal), new_g, neighbor))

    return None


# Computes the A* path from the Thymio's position (x, y in map pixels) to the goal (row, col).
# Called from main.ipynb only at the start and after each local navigation, not at every frame.
# Returns a list of (x, y) points in map pixels, None if no path found
def plan_path(grid, x, y, goal):
    grid = grid.copy()
    start = (int(round(y)), int(round(x)))

    # Frees the safety margin around the robot and the goal: after an obstacle avoidance,
    # the robot stands inside the inflated margin of the obstacle and A* would find no path
    yy, xx = np.ogrid[:MAP_HEIGHT, :MAP_WIDTH]
    for r, c in (start, goal):
        grid[(yy - r) ** 2 + (xx - c) ** 2 <= ROBOT_RADIUS_PX ** 2] = False

    path = a_star(grid, start, goal)
    if path is None:
        return None
    return [(c, r) for r, c in path]     # (row, col) -> (x, y)


# Method used to fetch the actual position and its angle, the final position and the predicted path
def fetch_status_cv(image):
    # Condition for fetching the status data: needs a video frame
    if image is None:
        raise ValueError("Image not found")

    # Calibration of camera
    rectified_image, markers, H = rectify_image(image)

    # Definiton of colored points coordinates
    mask_red, mask_green, mask_blue = color_masks(rectified_image)
    red_position = find_position(mask_red)
    green_position = find_position(mask_green)
    blue_position = find_position(mask_blue)

    # Case not allowing to define the actual position of the Thymio
    if red_position is None or green_position is None:
        raise ValueError("Thymio not found: red or green point not detected")
    # Case not allowing to define the final position
    if blue_position is None:
        raise ValueError("Goal not found: blue point not detected")

    # Thymio's position and angle definition
    x = (red_position[0] + green_position[0]) / 2
    y = (red_position[1] + green_position[1]) / 2
    theta = float(np.arctan2(green_position[1] - red_position[1],
                             green_position[0] - red_position[0]))

    # Generation of occupancy grid to know where the obstacles stand
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_blue)
    obstacle_mask = obstacle_map(rectified_image, color_mask, markers, H)
    # Forbidden zone = obstacles + margin + borders. Also displayed in orange by the GUI ("margin")
    occupancy_grid = occupancy_grid_from(obstacle_mask)

    # Goal cell (row, col). The path itself is computed by plan_path(), called from main.ipynb only when needed
    goal = (blue_position[1], blue_position[0])

    return [x, y, theta, occupancy_grid, goal, obstacle_positions(obstacle_mask)]
