import time

TURN_SPEED = 150          # spin speed
SPEED = 150               # forward speed
SEE_THR = 1000            # a sensor above this = obstacle seen -> start turning
CLEAR_THR = 300           # all sensors below this = nothing seen (lower than SEE_THR = hysteresis)
FORWARD_TIME = 0.6        # s driven forward after the obstacle is no longer seen
TURN_TIMEOUT = 4.0        # s, safety: give control back if the robot spins for too long


def new_local_nav_memory():
    return {"state": "OFF", "side": None, "t_state": 0.0}


def reset(mem):
    mem["state"] = "OFF"


def _side_of_obstacle(p):
    left = p[0] + p[1]
    right = p[3] + p[4]
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "left"        # obstacle exactly in front: arbitrary choice


def _spin(side):
    # obstacle on the left -> spin right, and the opposite
    return (TURN_SPEED, -TURN_SPEED) if side == "left" else (-TURN_SPEED, TURN_SPEED)


def local_nav_step(p, mem, alpha=None):
    """p = prox_horizontal. Returns (active, left, right).
    active = False -> use global navigation."""
    now = time.time()
    seen = max(p)

    # ---------------- OFF: global navigation ----------------
    if mem["state"] == "OFF":
        if seen <= SEE_THR:
            return False, 0, 0
        mem["side"] = _side_of_obstacle(p)       # chosen once, kept during the whole turn
        mem["state"], mem["t_state"] = "TURN", now

    # ---------------- TURN: spin away until nothing is seen ----------------
    if mem["state"] == "TURN":
        if seen < CLEAR_THR:
            mem["state"], mem["t_state"] = "FORWARD", now
        elif now - mem["t_state"] > TURN_TIMEOUT:
            mem["state"] = "OFF"
            return False, 0, 0
        else:
            return True, *_spin(mem["side"])

    # ---------------- FORWARD: drive straight for FORWARD_TIME ----------------
    if mem["state"] == "FORWARD":
        if seen > SEE_THR:                        # something new ahead: turn again
            mem["side"] = _side_of_obstacle(p)
            mem["state"], mem["t_state"] = "TURN", now
            return True, *_spin(mem["side"])
        if now - mem["t_state"] >= FORWARD_TIME:
            mem["state"] = "OFF"                  # back to global navigation
            return False, 0, 0
        return True, SPEED, SPEED

    return False, 0, 0