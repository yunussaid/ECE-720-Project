import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-graphical mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Default PyBullet data path

# Set Gravity
p.setGravity(0, 0, -9.8)

# Load Ground Plane
planeId = p.loadURDF("plane.urdf")

# Load the LeArm Robot
robot_start_pos = [0, 0, 0]  # Adjust if necessary
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

robotId = p.loadURDF("learm_description/learm.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)

# Simulate
for i in range(10000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # 240Hz simulation step

# Disconnect from PyBullet
p.disconnect()
