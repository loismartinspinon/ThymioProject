import cv2
import numpy as np
import heapq

# Size of the rectified map (pixels)
MAP_WIDTH = 205
MAP_HEIGHT = 155

# HSV color ranges
LOWER_RED1, UPPER_RED1 = np.array([0, 100, 100]), np.array([10, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([160, 100, 100]), np.array([180, 255, 255])
LOWER_GREEN, UPPER_GREEN = np.array([35, 80, 80]), np.array([90, 255, 255])
LOWER_BLUE, UPPER_BLUE = np.array([100, 100, 100]), np.array([130, 255, 255])

# Obstacles
GREY_THRESHOLD = 80        # grey level <= threshold -> obstacle
MIN_OBSTACLE_AREA = 20     # blobs smaller than this (pixels) are noise

# Half-size of the Thymio in pixels of the rectified map + safety margin
ROBOT_RADIUS_PX = 12

nest_position = []

def detect_qr_codes(image):
    # QRCodeDetectorAruco is more robust: the standard QRCodeDetector
    # sometimes finds only 3 of the 4 QR codes (e.g. ThymioSetup2/3)
    points = None
    for detector in (cv2.QRCodeDetectorAruco(), cv2.QRCodeDetector()):
        retval, points = detector.detectMulti(image)
        if retval and points is not None and len(points) == 4:
            return points
    return points


def order_corners(p):
    # Order a QR code's corners geometrically: TL, TR, BR, BL
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).flatten()
    return np.array([
        p[np.argmin(s)],  # top left
        p[np.argmin(d)],  # top right
        p[np.argmax(s)],  # bottom right
        p[np.argmax(d)]   # bottom left
    ], dtype=np.float32)


def rectify_image(image):
    points = detect_qr_codes(image)

    n_found = 0 if points is None else len(points)
    if n_found != 4:
        raise ValueError(f"Exactly 4 QR codes are required, found {n_found}")

    markers = np.array([order_corners(m) for m in points.reshape(4, 4, 2)])

    # Sort QR codes top / bottom, then left / right
    centers = markers.mean(axis=1)
    idx = np.argsort(centers[:, 1])
    top = sorted(markers[idx[:2]], key=lambda p: p[:, 0].mean())
    bottom = sorted(markers[idx[2:]], key=lambda p: p[:, 0].mean())

    # Take ONLY the 4 external corners
    src = np.array([
        top[0][0],       # top left QR: top left corner
        top[1][1],       # top right QR: top right corner
        bottom[1][2],    # bottom right QR: bottom right corner
        bottom[0][3]     # bottom left QR: bottom left corner
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [MAP_WIDTH - 1, 0],
        [MAP_WIDTH - 1, MAP_HEIGHT - 1],
        [0, MAP_HEIGHT - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    rectified_image = cv2.warpPerspective(image, H, (MAP_WIDTH, MAP_HEIGHT))

    return rectified_image, markers, H

def color_masks(rectified_image):
    hsv = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
        cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    )
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    return mask_red, mask_green, mask_blue


def find_position(mask):
    # Centroid (x, y) of the largest blob of the mask, None if nothing found
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    if not valid_contours:
        return None

    M = cv2.moments(max(valid_contours, key=cv2.contourArea))
    if M["m00"] == 0:
        return None

    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def obstacle_map(rectified_image, color_mask, markers, H):
    grey_image = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)

    # Dark pixels, except the colored points
    color_mask = cv2.dilate(color_mask, np.ones((3, 3), np.uint8), iterations=1)
    obstacle_mask = (grey_image <= GREY_THRESHOLD) & (color_mask == 0)

    # Remove the 4 QR codes: project their corners into the rectified image with H
    qr_mask = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
    qr_corners_rectified = cv2.perspectiveTransform(markers.reshape(-1, 1, 2), H).reshape(4, 4, 2)
    for qr in qr_corners_rectified:
        cv2.fillConvexPoly(qr_mask, np.round(qr).astype(np.int32), 255)
    qr_mask = cv2.dilate(qr_mask, np.ones((5, 5), np.uint8), iterations=1)
    obstacle_mask &= (qr_mask == 0)

    # Remove the small noise blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        obstacle_mask.astype(np.uint8), connectivity=8
    )
    for label in range(1, n_labels):  # label 0 = background
        if stats[label, cv2.CC_STAT_AREA] < MIN_OBSTACLE_AREA:
            obstacle_mask[labels == label] = False

    return obstacle_mask


def occupancy_grid_from(obstacle_mask):
    # Grow every obstacle (and the map border) by the robot radius:
    # the planning is then done for the centre of the robot
    kernel_size = 2 * ROBOT_RADIUS_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated = cv2.dilate(obstacle_mask.astype(np.uint8), kernel)

    inflated[:ROBOT_RADIUS_PX, :] = 1
    inflated[-ROBOT_RADIUS_PX:, :] = 1
    inflated[:, :ROBOT_RADIUS_PX] = 1
    inflated[:, -ROBOT_RADIUS_PX:] = 1

    return inflated.astype(bool)   # True = forbidden cell

def heuristic(a, b):
    # Octile distance: exact cost on an 8-connected grid without obstacles
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)


def a_star(grid, start, goal):
    rows, cols = grid.shape

    # 8-connectivity: (d_row, d_col, cost)
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
             (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

    open_set = [(heuristic(start, goal), 0.0, start)]   # (f, g, cell)
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

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

def fetch_status_cv(image):
    if image is None:
        raise ValueError("Image not found")

    # 1. Camera calibration
    rectified_image, markers, H = rectify_image(image)

    # 2. Colored points
    mask_red, mask_green, mask_blue = color_masks(rectified_image)
    red_position = find_position(mask_red)
    green_position = find_position(mask_green)
    blue_position = find_position(mask_blue)

    if red_position is None or green_position is None:
        raise ValueError("Thymio not found: red or green point not detected")
    if blue_position is None:
        raise ValueError("Goal not found: blue point not detected")

    # 3. Thymio position and angle
    x = (red_position[0] + green_position[0]) / 2
    y = (red_position[1] + green_position[1]) / 2
    theta = float(np.arctan2(green_position[1] - red_position[1],
                             green_position[0] - red_position[0]))

    # 4. Obstacles
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_blue)
    obstacle_mask = obstacle_map(rectified_image, color_mask, markers, H)
    occupancy_grid = occupancy_grid_from(obstacle_mask)

    # 5. A* path (grid cells are (row, col) = (y, x))
    start = (int(round(y)), int(round(x)))
    goal = (blue_position[1], blue_position[0])

    # Start / goal inside an inflated obstacle: force them to free
    occupancy_grid[start] = False
    occupancy_grid[goal] = False

    path = a_star(occupancy_grid, start, goal)

    if path is None:
        next_position = None
    else:
        # Next cell of the path (path[0] is the start itself)
        row, col = path[1] if len(path) > 1 else path[0]
        next_position = (col, row)   # (row, col) -> (x, y)

    return [x, y, theta, path]