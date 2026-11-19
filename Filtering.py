"""
Filtering: estimation of the Thymio pose (x, y, theta) (from Filtering.ipynb).

- odometry(): pose predicted from the wheel speeds
- kalman():   prediction with the odometry, then correction with the camera pose
- wrap_to_pi(): angle brought back into ]-pi, pi] (also used by MotionController)

Initialisation (in main): state = camera pose of the first frame, P = R.copy()
The test with the simulated robot stays in Filtering.ipynb.
"""
#Importation 
import math
import numpy as np
from math import sin, cos, atan2

#Global variable 
ROT_SPEED_TO_M_PER_S = 0.0003375  # needs to be calibrated
WHEEL_BASE = 0.095 # needs to be calibrated


# constants, to tune
SIGMA_D2 = 1e-4      # variance added per meter driven
SIGMA_TH2 = 1e-3     # variance added per radian turned
Q_MIN = 1e-8         # small floor so P never becomes exactly 0
R = np.diag([1e-4, 1e-4, 1e-3])   # camera uncertainty
H = np.eye(3)


def wrap_to_pi(a):
    return atan2(sin(a), cos(a))


def odometry(rot_speed_left_motor, rot_speed_right_motor, X_position, Y_position, angle_position_rad, delta_time):
    
    #Linear speed
    linear_speed_left = rot_speed_left_motor * ROT_SPEED_TO_M_PER_S
    linear_speed_right = rot_speed_right_motor * ROT_SPEED_TO_M_PER_S
    linear_speed = (linear_speed_right + linear_speed_left) / 2 
    
    #Angular speed
    angular_speed = (linear_speed_right - linear_speed_left) / WHEEL_BASE  # rad/s

    angle_mid = angle_position_rad + angular_speed * delta_time / 2 #middle point to make an etimation of 2 segment
     
    X_new = X_position + linear_speed * math.cos(angle_mid) * delta_time
    Y_new = Y_position + linear_speed * math.sin(angle_mid) * delta_time
    angle_new = angle_position_rad + angular_speed * delta_time

    return X_new, Y_new, angle_new


def kalman(state, P, speed_left, speed_right, dt, camera_pose):

    # ---------- PREDICTION ----------
    X, Y, theta = state
    state_pred = np.array(odometry(speed_left, speed_right, X, Y, theta, dt))
    state_pred[2] = wrap_to_pi(state_pred[2])

    v = (speed_right + speed_left) * ROT_SPEED_TO_M_PER_S / 2
    omega = (speed_right - speed_left) * ROT_SPEED_TO_M_PER_S / WHEEL_BASE
    theta_mid = theta + omega * dt / 2

    # F: how an error in the previous state propagates into the new state
    F = np.array([[1, 0, -v * sin(theta_mid) * dt],
                  [0, 1,  v * cos(theta_mid) * dt],
                  [0, 0,  1]])

    # Q: uncertainty added by this step, proportional to the motion
    q_xy = SIGMA_D2 * abs(v * dt) + Q_MIN
    q_th = SIGMA_TH2 * abs(omega * dt) + Q_MIN
    Q = np.diag([q_xy, q_xy, q_th])

    P_pred = F @ P @ F.T + Q

    if camera_pose is None:
        return state_pred, P_pred          # odometry only

    # ---------- UPDATE ----------
    innovation = camera_pose - H @ state_pred
    innovation[2] = wrap_to_pi(innovation[2])

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state_new = state_pred + K @ innovation
    state_new[2] = wrap_to_pi(state_new[2])
    P_new = (np.eye(3) - K @ H) @ P_pred
    return state_new, P_new
