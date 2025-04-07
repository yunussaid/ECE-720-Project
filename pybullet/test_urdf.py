import pybullet as pyB
import pybullet_data
import time

# Connect to PyBullet
physicsClient = pyB.connect(pyB.GUI)  # Use pyB.DIRECT for non-graphical mode
pyB.setAdditionalSearchPath(pybullet_data.getDataPath())  # Default PyBullet data path

# Set Gravity
pyB.setGravity(0, 0, -9.8)

# Load Ground Plane
planeId = pyB.loadURDF("plane.urdf")

# Load the LeArm Robot
robot_start_pos = [0, 0, 0]
robot_start_orientation = pyB.getQuaternionFromEuler([0, 0, 0])
robotId = pyB.loadURDF("learm_description/learm.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)

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
disp_net_coords = False

if mode == "all":
    # Simulate Mobility for All Joints
    sliders = []
    num_joints = pyB.getNumJoints(robotId)

    for joint in range(num_joints):
        info = pyB.getJointInfo(robotId, joint)
        joint_name = info[1].decode("utf-8")
        sliders.append(pyB.addUserDebugParameter(joint_name, -5, 5, 0))

    while True:
        for joint in range(num_joints):
            target_pos = pyB.readUserDebugParameter(sliders[joint])
            pyB.setJointMotorControl2(robotId, joint, pyB.POSITION_CONTROL, targetPosition=target_pos)
        pyB.stepSimulation()
        time.sleep(0.01)

elif mode == "prj":
    # Simulate Joint Mobilities According to Project Specifications
    sliders = []
    sliders.append(pyB.addUserDebugParameter("base_joint", -1.57, 1.57, 0))
    sliders.append(pyB.addUserDebugParameter("shoulder_joint", 0, 1.57, 0))
    # sliders.append(pyB.addUserDebugParameter("ball_x", -10, 10, 0))
    # sliders.append(pyB.addUserDebugParameter("ball_y", -10, 10, 0))
    # sliders.append(pyB.addUserDebugParameter("ball_z", -10, 10, 0))

    while True:
        base_joint_target_pos = pyB.readUserDebugParameter(sliders[joint_idx["base_joint"]])
        pyB.setJointMotorControl2(robotId, joint_idx["base_joint"], pyB.POSITION_CONTROL, targetPosition=base_joint_target_pos)

        shoulder_joint_target_pos = pyB.readUserDebugParameter(sliders[joint_idx["shoulder_joint"]])
        pyB.setJointMotorControl2(robotId, joint_idx["shoulder_joint"], pyB.POSITION_CONTROL, targetPosition=shoulder_joint_target_pos)
        pyB.setJointMotorControl2(robotId, joint_idx["wrist_joint"], pyB.POSITION_CONTROL, targetPosition=shoulder_joint_target_pos)
        
        if disp_net_coords:
            net_id = joint_idx["net_joint"]  # Get the joint index for the net link
            net_pos, _ = pyB.getLinkState(robotId, net_id)[:2]  # Get the position of the net link (no need for orientation)
            print(f"Net Position: [{round(net_pos[0], 2)}, {round(net_pos[1], 2)}, {round(net_pos[2], 2)}]")
            pyB.addUserDebugPoints([list(net_pos)], pointColorsRGB=[[1, 0, 0]], pointSize=5)
        
        # ball_target_x = pyB.readUserDebugParameter(sliders[2])
        # ball_target_y = pyB.readUserDebugParameter(sliders[3])
        # ball_target_z = pyB.readUserDebugParameter(sliders[4])

        pyB.stepSimulation()
        time.sleep(0.01)

# Disconnect from PyBullet
pyB.disconnect()
