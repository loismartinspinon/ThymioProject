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
# THETA_IN_DEGREES = False     # True if theta0 is in degrees
# THETA_CONVENTION = "math"    # "math"  : 0 = +x axis, counter-clockwise positive (standard)
                             # "robot" : 0 = +y axis, clockwise positive (as in this code)

client = ClientAsync()

# ---------- State of the robot ----------
# current_position = [0.0, 0.0]
# current_angle = 0.0          # 0 = towards +y, clockwise positive


# =============================================================================
# Useful functions
# =============================================================================

def angle_to_target(pos, target):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    return np.arctan2(dy, dx)


async def set_motors(node, left, right):
    await node.set_variables({
        "motor.left.target": [int(left)],
        "motor.right.target": [int(right)],
    })


# =============================================================================
# Moves
# =============================================================================

async def turn(node, deltaTheta):
    diff = deltaTheta

    if abs(diff) > min_angle:
        sign = 1 if diff > 0 else -1
        await set_motors(node, -sign * speed_turn, sign * speed_turn)
        await client.sleep(abs(diff) / omega)
        await set_motors(node, 0, 0)
        return (-sign * speed_turn, sign * speed_turn, abs(diff) / omega)
    else:
        return None



async def go_forward(node, distance):

    await set_motors(node, speed_forward, speed_forward)
    await client.sleep(distance / v_forward)
    await set_motors(node, 0, 0)
    return (speed_forward, speed_forward, distance / v_forward)

async def move_to(pose, target): # a changer
    x, y, theta = pose
    dist = np.hypot(target[0] - x,
                    target[1] - y)
    if dist < min_distance:      # point already reached, avoids arctan2(0, 0)
        return []
        
    deltaTheta = wrap_to_pi(angle_to_target((x,y), target) - theta)

    node = await client.wait_for_node()
    await node.lock()
    moves = []
    try:
        rotation = await turn(node, deltaTheta)
        if rotation is not None:
            moves.append(rotation)
        moves.append(await go_forward(node, dist))
    finally:
        await set_motors(node, 0, 0)
        await node.unlock()
    return moves


    await turn(node, target)
    await go_forward(node, target)