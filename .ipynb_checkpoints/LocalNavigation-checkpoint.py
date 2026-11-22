import time
import numpy as np
import MotionController as mc
from tdmclient import ClientAsync

# ---------- constants ----------
SPEED = 100
TURN_SPEED = 120
FRONT_THR = 1000
FRONT_CLEAR = 300
SIDE_SEEN = 300
EXTRA_TIME = 0.7
TURN_TIMEOUT = 4.0
STEP = 0.05

def new_local_nav_memory():
    return {"state": "OFF", "side": None, "t_state": 0.0, "t_lost": None}

def _turn(side):
    # obstacle on the left -> spin right, and the opposite
    return (TURN_SPEED, -TURN_SPEED) if side == "left" else (-TURN_SPEED, TURN_SPEED)

def local_nav_step(p, mem):
    """Call at every loop with p = prox_horizontal.
    Returns (active, left, right). active = False means: use global navigation."""
    now = time.time()
    front = max(p[1], p[2], p[3])

    if mem["state"] == "OFF":
        if front <= FRONT_THR:
            return False, 0, 0
        mem["side"] = "left" if (p[0] + p[1]) >= (p[3] + p[4]) else "right"
        mem["state"], mem["t_state"] = "TURN", now

    side_val = p[0] if mem["side"] == "left" else p[4]

    if mem["state"] == "TURN":
        parallel = front < FRONT_CLEAR and side_val > SIDE_SEEN
        if parallel or now - mem["t_state"] > TURN_TIMEOUT:
            mem["state"], mem["t_lost"] = "PASS", None
            return (True, SPEED, SPEED)
        return (True, *_turn(mem["side"]))

    # state == "PASS"
    if front > FRONT_THR:                         # corner or new obstacle
        mem["state"], mem["t_state"] = "TURN", now
        return (True, *_turn(mem["side"]))
    if side_val < SIDE_SEEN:
        if mem["t_lost"] is None:
            mem["t_lost"] = now
        elif now - mem["t_lost"] > EXTRA_TIME:
            mem["state"] = "OFF"                  # done, back to global
            return False, 0, 0
    else:
        mem["t_lost"] = None
    return True, SPEED, SPEED

async def avoid(mem):
    node = await mc.client.wait_for_node()
    await node.lock()
    moves = []
    try:
        await node.wait_for_variables({"prox.horizontal"})
        active, left, right = local_nav_step(list(node.v.prox.horizontal), mem)
        while active:
            await mc.set_motors(node, left, right)
            await mc.client.sleep(STEP)
            moves.append((left, right, STEP))
            active, left, right = local_nav_step(list(node.v.prox.horizontal), mem)
    finally:
        await mc.set_motors(node, 0, 0)
        await node.unlock()
    return moves