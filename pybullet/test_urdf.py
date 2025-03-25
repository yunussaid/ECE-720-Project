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
robot_start_pos = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("learm_description/learm.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)

# Define Joint Indices Dictionary
joint_idx = {
    "base_joint": 0,
    "shoulder_joint": 1,
    "elbow_joint": 2,
    "wrist_joint": 3,
    "net_joint": 4
}

# Set the Mode
mode = "prj"  # "prj" or "all"

if mode == "all":
    # Simulate Mobility for All Joints
    sliders = []
    num_joints = p.getNumJoints(robotId)

    for joint in range(num_joints):
        info = p.getJointInfo(robotId, joint)
        joint_name = info[1].decode("utf-8")
        sliders.append(p.addUserDebugParameter(joint_name, -1.57, 1.57, 0))

    while True:
        for joint in range(num_joints):
            target_pos = p.readUserDebugParameter(sliders[joint])
            p.setJointMotorControl2(robotId, joint, p.POSITION_CONTROL, targetPosition=target_pos)
        p.stepSimulation()
        time.sleep(0.01)

elif mode == "prj":
    # Simulate Joint Mobilities According to Project Specifications
    sliders = []
    sliders.append(p.addUserDebugParameter("base_joint", -1.57, 1.57, 0))
    sliders.append(p.addUserDebugParameter("shoulder_joint", 0, 1.57, 0))

    while True:
        base_joint_target_pos = p.readUserDebugParameter(sliders[joint_idx["base_joint"]])
        p.setJointMotorControl2(robotId, joint_idx["base_joint"], p.POSITION_CONTROL, targetPosition=base_joint_target_pos)

        shoulder_joint_target_pos = p.readUserDebugParameter(sliders[joint_idx["shoulder_joint"]])
        p.setJointMotorControl2(robotId, joint_idx["shoulder_joint"], p.POSITION_CONTROL, targetPosition=shoulder_joint_target_pos)
        p.setJointMotorControl2(robotId, joint_idx["wrist_joint"], p.POSITION_CONTROL, targetPosition=shoulder_joint_target_pos)

        p.stepSimulation()
        time.sleep(0.01)

# Disconnect from PyBullet
p.disconnect()
