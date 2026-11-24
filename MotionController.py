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

k_rho, k_alpha, k_beta = 1.0, 3.0, 0.0
assert k_rho > 0 and k_beta <= 0 and k_alpha - k_rho > 0

v_max = 4.0
w_max = 0.7
motor_max = 255
l = 11.7 / 2
speed_factor = speed_forward / v_forward
dt = 0.05

# ---------- Format of theta0 given by the vision (TO CHECK) ----------
THETA_IN_DEGREES = False     # True if theta0 is in degrees
THETA_CONVENTION = "math"    # "math"  : 0 = +x axis, counter-clockwise positive (standard)
                             # "robot" : 0 = +y axis, clockwise positive (as in this code)

client = ClientAsync()

# ---------- State of the robot ----------
current_position = [0.0, 0.0]
next_position = [0.0, 0.0]
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

def astolfi_controller(x, y, theta, goal):

    delta_x = goal[0] - x
    delta_y = goal[1] - y

    rho = np.hypot(delta_x, delta_y)
    alpha = wrap_to_pi(-theta + np.arctan2(delta_y, delta_x))
    beta = wrap_to_pi(-theta - alpha)

    v = k_rho * rho
    w = k_alpha * alpha + k_beta * beta

    return v, w

def v_w_to_motors(v, w):
    v = np.clip(v, -v_max, v_max)
    w = np.clip(w, -w_max, w_max)

    v_right = v + l * w      # cm/s
    v_left = v - l * w

    speed_right = np.clip(v_right * speed_factor, -motor_max, motor_max)
    speed_left = np.clip(v_left * speed_factor, -motor_max, motor_max)
    return speed_left, speed_right

def robot_to_math(a):
    return wrap_to_pi(np.pi / 2 - a)

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

async def move_to_deux(node, target, is_last):
    x, y = current_position[0], current_position[1]
    theta = convert_theta(current_angle)
    if is_last :
        rayon = 2
    else :
        rayon = 5
    while rho > rayon :
        v, w = astolfi_controller(x, y, theta, target)
        speed_left, speed_right = v_w_to_motors(v,w)
        await set_motors(node, speed_left, speed_right)
        await client.sleep(dt)

        v_applied = (speed_right + speed_left) / 2 / speed_factor      # cm/s
        w_applied = (speed_right - speed_left) / 2 / speed_factor / l  # rad/s
        
        theta += w_applied * dt                         # New theta
        x += v_applied * np.cos(theta) * dt   # New x
        y += v_applied * np.sin(theta) * dt   # New y

        rho = np.hypot(target[0] - x, target[1] - y)

    current_position[0], current_position[1] = x, y
    current_angle = robot_to_math(theta)



async def follow_path(node, path):
    for i, point in enumerate(path):
        print(f"Point {i + 1}/{len(path)} : {point}")
        await move_to_deux(node, point, i == len(path) - 1)


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