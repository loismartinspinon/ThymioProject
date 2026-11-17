"""
Motion controller of the Thymio (from Barth_MotionController.ipynb).

run(data) -> follow_path -> move_to: for each point of the path, the robot turns on itself
towards the point, then goes straight to it. Both moves are open loop (duration =
angle / omega and distance / v_forward), with omega and v_forward calibrated in
Barth_MotionController.ipynb.

Usage (main.ipynb):
    import MotionController as mc
    await mc.run([x0, y0, theta0, path])    # cm, theta0 in the THETA_CONVENTION below
run() locks the Thymio itself and always stops the motors at the end.

Requires Thymio Suite (Thymio Device Manager) running and the Thymio connected.
Do not also use tdmclient.notebook.start(): it already locks the Thymio, and node.lock()
below would then fail with "Node lock error (current status: busy)".
"""
import numpy as np
from tdmclient import ClientAsync

from Filtering import wrap_to_pi   # angle brought back into ]-pi, pi] (replaces normalize)


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

client = ClientAsync()

# ---------- State of the robot ----------
current_position = [0.0, 0.0]
current_angle = 0.0          # 0 = towards +y, clockwise positive


# =============================================================================
# Useful functions
# =============================================================================

def convert_theta(theta):
    """Converts the received theta0 to the robot convention (0 = +y, clockwise)."""
    if THETA_IN_DEGREES:
        theta = np.radians(theta)
    if THETA_CONVENTION == "math":
        theta = np.pi / 2 - theta
    return wrap_to_pi(theta)


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
    diff = wrap_to_pi(next_angle - current_angle)

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