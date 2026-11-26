import math
import time

SPEED = 150
TURN_SPEED = 300
FRONT_THR = 1500          # obstacle too close ahead -> enter local nav (to tune)
FRONT_CLEAR = 200         # front considered clear below this
SIDE_SEEN = 300           # side sensor sees the object above this
SIDE_TARGET = 800        # side sensor value to keep while following (to tune)
FOLLOW_GAIN = 0.1         # to tune
MAX_ERR = 400
ALPHA_TOL = math.radians(15)   # "same angle": exit when |alpha| is below this
TURN_TIMEOUT = 4.0

def new_local_nav_memory():
    return {"state": "OFF", "side": None, "t_state": 0.0}

def _spin(side):
    # object on the left -> spin right, and the opposite
    return (TURN_SPEED, -TURN_SPEED) if side == "left" else (-TURN_SPEED, TURN_SPEED)

def local_nav_step(p, mem, alpha):
    """p = prox_horizontal, alpha = heading error to the target (rad).
    Returns (active, left, right). active = False -> use global navigation."""
    now = time.time()
    front = max(p[1], p[2], p[3])

    if mem["state"] == "OFF":
        if front <= FRONT_THR:
            return False, 0, 0
        mem["side"] = "left" if (p[0] + p[1]) >= (p[3] + p[4]) else "right"
        mem["state"], mem["t_state"] = "TURN", now

    side = mem["side"]
    side_val = p[0] if side == "left" else p[4]

    # --- TURN: spin until parallel to the object ---
    if mem["state"] == "TURN":
        parallel = front < FRONT_CLEAR and side_val > SIDE_SEEN
        if parallel or now - mem["t_state"] > TURN_TIMEOUT:
            mem["state"] = "FOLLOW"
        else:
            return (True, *_spin(side))

    # --- FOLLOW: drive along the object ---
    if front > FRONT_THR:                       # corner or new obstacle ahead
        mem["state"], mem["t_state"] = "TURN", now
        return (True, *_spin(side))

    if front < FRONT_CLEAR and abs(alpha) < ALPHA_TOL:
        mem["state"] = "OFF"                    # facing the target: back to global
        return False, 0, 0

    # keep the side sensor near SIDE_TARGET; if the object is lost, curve toward it
    err = max(-MAX_ERR, min(MAX_ERR, side_val - SIDE_TARGET))
    s = 1 if side == "left" else -1
    left = SPEED + s * FOLLOW_GAIN * err
    right = SPEED - s * FOLLOW_GAIN * err
    return True, left, right