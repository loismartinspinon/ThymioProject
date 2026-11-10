# ThymioProject

# Autonomous Point-to-Point Navigation

## 1. Project Overview

The objective of this project is to develop an autonomous mobile robot that can navigate from a starting point **A** to a destination **B**, while avoiding obstacles along its path.

The robot will operate in a simple environment with a white floor and a few static obstacles. A camera placed above the environment will track the robot's position.

The project is divided into five main components: **Vision, Global Navigation, Local Navigation, Motion Controller, and Filtering**.

## 2. System Components

### 2.1 Vision

**Objective:** Determine the robot's position and orientation.

- Use an overhead camera to detect the robot.
- Identify the robot using colored markers.
- Extract its position (x, y) from the camera image.
- Estimate its orientation (θ) based on the markers.
- Continuously provide the robot's position to the other components.

**Output:** Robot position and orientation: (x, y, θ).

### 2.2 Global Navigation

**Objective:** Find a path from point A to point B.

- Define a simple map of the environment.
- Represent the obstacles on the map.
- Calculate a collision-free path from the starting point to the destination.
- Divide the path into a sequence of intermediate points (waypoints).

**Output:** A planned path from A to B.

### 2.3 Local Navigation

**Objective:** Avoid obstacles that are too close to the robot.

- Use the robot's proximity sensors to detect nearby obstacles.
- If an obstacle is detected within a certain distance, temporarily modify the robot's movement to avoid it.
- Once the obstacle is avoided, resume following the global path.

**Output:** A safe direction or movement adjustment when an obstacle is detected.

### 2.4 Motion Controller

**Objective:** Control the robot's motors so that it follows the planned path.

- Calculate the direction of the next waypoint.
- Compare this direction with the robot's current orientation.
- Adjust the left and right motor speeds to steer the robot towards the waypoint.
- Reduce the speed when the robot needs to turn.
- Stop the robot when it reaches point B.

A simple proportional controller (P-controller) can be used to control the robot's orientation and movement.

**Output:** Left and right motor speed commands.

### 2.5 Filtering

**Objective:** Obtain a more accurate estimate of the robot's position.

- Obtain the robot's position from the camera (Vision).
- Obtain an estimated position from the robot's odometry (wheel movements).
- Combine both sources using a Kalman Filter.
- Use the filtered position and orientation instead of relying only on raw measurements.

**Output:** A filtered estimate of the robot's position and orientation: (x, y, θ).

## 3. Expected Behaviour

The final system should perform the following sequence:

1. The robot starts at point A.
2. Vision detects its initial position and orientation.
3. Global Navigation calculates a path to point B.
4. The Motion Controller makes the robot follow the planned path.
5. Local Navigation detects and avoids obstacles that are too close.
6. Filtering continuously improves the robot's position estimate using camera measurements and odometry.
7. The robot reaches point B and stops.

## 4. Environment and Setup

To keep the project simple, the environment will consist of:

- A flat white surface.
- One starting point (A).
- One destination point (B).
- A small number of static obstacles.
- A mobile robot equipped with proximity sensors.
- An overhead camera.
- Colored markers on the robot for visual detection.

The project will focus on demonstrating the integration of the five components rather than building a complex navigation system.
