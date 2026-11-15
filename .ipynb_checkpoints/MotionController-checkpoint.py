"""
Motion controller of the Thymio (from Barth_MotionController.ipynb).

The robot follows a path point by point: for each point it turns on itself towards the
point, then goes straight to it. Both moves are open loop (duration = angle / omega and
distance / v_forward), with omega and v_forward calibrated with calibration() below.

Usage:
    from Barth_MotionController import run
    await run([x0, y0, theta0, path])          # in a notebook (main.ipynb)

    python Barth_MotionController.py           -> runs the test path TEST_DATA

Requires Thymio Suite (Thymio Device Manager) running and the Thymio connected.
Do not also use tdmclient.notebook.start(): it already locks the Thymio, and node.lock()
below would then fail with "Node lock error (current status: busy)".
"""
import numpy as np
from tdmclient import ClientAsync


# =============================================================================
# Parameters to calibrate
# =============================================================================

speed_turn = 100
speed_forward = 150
omega = 0.628                # rad/s at speed_turn
v_forward = 4.833            # cm/s at speed_forward
min_angle = np.radians(1)    # below this angle, the robot does not turn
min_distance = 0.5           # below this distance (cm), the point is ignored

# ---------- Format of theta0 given by the vision (TO CHECK) ----------
THETA_IN_DEGREES = False     # True if theta0 is in degrees
THETA_CONVENTION = "math"    # "math"  : 0 = +x axis, counter-clockwise positive (standard)
                             # "robot" : 0 = +y axis, clockwise positive (as in this code)

# ---------- Calibration tests ----------
duree = 5          # duration of each test (s)
nb_essais = 3      # number of tests
pause = 6          # time between two tests (s)

client = ClientAsync()

# ---------- State of the robot ----------
current_position = [0.0, 0.0]
current_angle = 0.0          # 0 = towards +y, clockwise positive


# =============================================================================
# Useful functions
# =============================================================================

def normalize(angle):
    """Brings the angle back into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def convert_theta(theta):
    """Converts the received theta0 to the robot convention (0 = +y, clockwise)."""
    if THETA_IN_DEGREES:
        theta = np.radians(theta)
    if THETA_CONVENTION == "math":
        theta = np.pi / 2 - theta
    return normalize(theta)


def angle_to_target(pos, target):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    return np.arctan2(dx, dy)


async def set_motors(node, left, right):
    await node.set_variables({
        "motor.left.target": [int(left)],
        "motor.right.target": [int(right)],
    })


# =============================================================================
# Moves
# =============================================================================

async def turn_to(node, target):
    global current_angle
    next_angle = angle_to_target(current_position, target)
    diff = normalize(next_angle - current_angle)

    if abs(diff) > min_angle:
        sign = 1 if diff > 0 else -1
        await set_motors(node, sign * speed_turn, -sign * speed_turn)
        await client.sleep(abs(diff) / omega)
        await set_motors(node, 0, 0)

    current_angle = next_angle


async def go_forward(node, target):
    dx = target[0] - current_position[0]
    dy = target[1] - current_position[1]
    distance = np.hypot(dx, dy)

    await set_motors(node, speed_forward, speed_forward)
    await client.sleep(distance / v_forward)
    await set_motors(node, 0, 0)

    current_position[0], current_position[1] = target[0], target[1]


async def move_to(node, target):
    dist = np.hypot(target[0] - current_position[0],
                    target[1] - current_position[1])
    if dist < min_distance:      # point already reached, avoids arctan2(0, 0)
        return
    await turn_to(node, target)
    await go_forward(node, target)


async def follow_path(node, path):
    for i, point in enumerate(path):
        print(f"Point {i + 1}/{len(path)} : {point}")
        await move_to(node, point)


# =============================================================================
# Main program
# =============================================================================

async def run(data):
    """data = [x0, y0, theta0, path] with path = [(x1, y1), ..., (xf, yf)]"""
    global current_angle
    x0, y0, theta0, path = data

    # State initialisation with the starting pose
    current_position[0], current_position[1] = x0, y0
    current_angle = convert_theta(theta0)

    node = await client.wait_for_node()
    await node.lock()
    try:
        await follow_path(node, path)
    finally:
        await set_motors(node, 0, 0)
        await node.unlock()


# Test path (in cm)
TEST_DATA = [0, 0, np.pi / 2, [
    (0, 20),     # 1 : straight
    (15, 20),    # 2 : to the right
    (15, 40),    # 3 : up
    (0, 40),     # 4 : to the left
    (-15, 30),   # 5 : diagonal
    (-15, 10),   # 6 : down
]]


# =============================================================================
# Calibration of the rotation and translation speeds of the wheels
# 1. await calibration("rotation") -> note the number of turns of each test
# 2. await calibration("avance")   -> note the distance (cm) of each test
# 3. compute_omega(turns) and compute_v_forward(distances) -> new omega / v_forward
# =============================================================================

async def essais(node, left, right):
    for i in range(nb_essais):
        print(f"Essai {i + 1}/{nb_essais} dans 3 s... (replace le robot)")
        await client.sleep(3)
        print("GO")
        await set_motors(node, left, right)
        await client.sleep(duree)
        await set_motors(node, 0, 0)
        print("STOP : note le résultat.")
        if i < nb_essais - 1:
            await client.sleep(pause)


async def calibration(test):
    node = await client.wait_for_node()
    await node.lock()
    try:
        if test == "rotation":
            await essais(node, speed_turn, -speed_turn)
        else:
            await essais(node, speed_forward, speed_forward)
    finally:
        await set_motors(node, 0, 0)
        await node.unlock()


def compute_omega(tours):
    """tours = number of turns measured at each rotation test -> omega (rad/s)"""
    omega = np.mean(tours) * 2 * np.pi / duree
    print(f"omega = {omega:.3f} rad/s")
    print(f"écart : {np.std(tours) * 2 * np.pi / duree:.3f} rad/s")
    return omega


def compute_v_forward(distances):
    """distances = distances (cm) measured at each forward test -> v_forward (cm/s)"""
    v = np.mean(distances) / duree
    print(f"v_forward = {v:.3f} cm/s")
    print(f"écart : {np.std(distances) / duree:.3f} cm/s")
    return v


if __name__ == "__main__":
    client.run_async_program(lambda: run(TEST_DATA))
