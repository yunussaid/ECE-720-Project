import numpy as np
import pybullet as pyB
import pybullet_data
import time
import gymnasium as gym
from gymnasium import spaces


class LearmArmEnv(gym.Env):
    def __init__(self, render=False):
        super(LearmArmEnv, self).__init__()

        self.render_mode = render
        self.steps_per_sec = 240.0
        self.time_step = 1.0 / self.steps_per_sec
        self.physics_client_id = None

        if self.render_mode:
            self.physics_client_id = pyB.connect(pyB.GUI)
        else:
            self.physics_client_id = pyB.connect(pyB.DIRECT)

        self.max_ep_length = 3 # seconds max per episode
        self.max_ep_steps = int(self.max_ep_length * self.steps_per_sec) # setps max per episode
        self.step_counter = 0

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.57, 0.0]),  # Base joint: [-1.57, 1.57], Shoulder joint: [0.0, 1.57]
            high=np.array([1.57, 1.57]),
            dtype=np.float32
        )
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)     # Observations: 2 joint positions + 2 joint velocities + 2 ball coordinates
        # self.observation_space = spaces.Box(
        #     low=np.array([-1.57, 0.0, -np.inf, -np.inf, -np.inf, -np.inf]),  # Base, shoulder positions; velocities remain unbounded
        #     high=np.array([1.57, 1.57, np.inf, np.inf, np.inf, np.inf]),  # Base, shoulder positions; velocities unbounded
        #     dtype=np.float32
        # )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)  # x-y coordinates for both net and ball

        self.base_joint_idx = None
        self.shoulder_joint_idx = None
        self.wrist_joint_idx = None
        self.net_joint_idx = None

        # Ball attributes
        self.ball_id = None
        self.ball_spawn_height = 5
        self.ball_radius = 0.2
        self.ball_noise_std = 0.1  # Adjust as needed

        self._setup_simulation()


    def _setup_simulation(self):
        pyB.resetDebugVisualizerCamera(
            cameraDistance=10,              # Zoom out by increasing distance
            cameraYaw=90,                   # Rotate view left/right
            cameraPitch=-85,                # Tilt up/down
            cameraTargetPosition=[0, 0, 0]  # Focus point (usually center of your robot)
        )
        pyB.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyB.setGravity(0, 0, -9.8)
        pyB.loadURDF("plane.urdf")

        robot_start_pos = [0, 0, 0]
        robot_start_orientation = pyB.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyB.loadURDF("learm_description/learm.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)

        _joint_names = ["base_joint", "shoulder_joint", "wrist_joint"]
        self.joint_indices = {
            pyB.getJointInfo(self.robot_id, i)[1].decode("utf-8"): i for i in range(pyB.getNumJoints(self.robot_id))
        }
        self.base_joint_idx = self.joint_indices["base_joint"]
        self.shoulder_joint_idx = self.joint_indices["shoulder_joint"]
        self.wrist_joint_idx = self.joint_indices["wrist_joint"]
        self.net_joint_idx = self.joint_indices["net_joint"]

        ball_visual = pyB.createVisualShape(pyB.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=[1, 0.5, 0, 1])
        ball_collision = pyB.createCollisionShape(pyB.GEOM_SPHERE, radius=self.ball_radius)

        r = np.random.uniform(1.0, 3.5)
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        spawn_x = r * np.cos(theta)
        spawn_y = r * np.sin(theta)
        spawn_z = self.ball_spawn_height

        self.ball_id = pyB.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=[spawn_x, spawn_y, spawn_z]
        )

        # self.draw_circle(1.0, [1, 0, 0])     # Inner red boundary
        # self.draw_circle(3.5, [0, 1, 0])     # Outer green boundary

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        pyB.resetSimulation()
        self._setup_simulation()

        # Reset joints
        pyB.resetJointState(self.robot_id, self.base_joint_idx, 0)
        pyB.resetJointState(self.robot_id, self.shoulder_joint_idx, 0)
        pyB.resetJointState(self.robot_id, self.wrist_joint_idx, 0)

        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1

        # Get current joint positions
        curr_base_angle = pyB.getJointState(self.robot_id, self.base_joint_idx)[0]
        curr_shoulder_angle = pyB.getJointState(self.robot_id, self.shoulder_joint_idx)[0]
        
        # Extract target joint angles from action and ensure they are within the valid joint limits
        target_base_angle, target_shoulder_angle = action
        target_base_angle = np.clip(target_base_angle, -1.57, 1.57)  # Base joint: [-1.57, 1.57]
        target_shoulder_angle = np.clip(target_shoulder_angle, 0.0, 1.57)  # Shoulder joint: [0.0, 1.57]

        # # Exponential moving average for smoothing target joint positions
        # alpha = 0.1  # Smoothing factor (between 0 and 1)
        # target_base_angle = alpha * target_base_angle + (1 - alpha) * curr_base_angle
        # target_shoulder_angle = alpha * target_shoulder_angle + (1 - alpha) * curr_shoulder_angle

        # Apply POSITION_CONTROL
        pyB.setJointMotorControl2(self.robot_id, self.base_joint_idx, pyB.POSITION_CONTROL, targetPosition=target_base_angle)
        pyB.setJointMotorControl2(self.robot_id, self.shoulder_joint_idx, pyB.POSITION_CONTROL, targetPosition=target_shoulder_angle)
        pyB.setJointMotorControl2(self.robot_id, self.wrist_joint_idx, pyB.POSITION_CONTROL, targetPosition=target_shoulder_angle)

        # Simulate one step
        pyB.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)

        # Get net and ball positions
        net_pos = pyB.getLinkState(self.robot_id, self.net_joint_idx)[0]
        ball_pos = pyB.getBasePositionAndOrientation(self.ball_id)[0]
        xy_dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(net_pos[:2]))

        # Default reward is negative distance in x-y plane
        reward = -xy_dist  # Base reward based on x-y distance

        # Success bonus for catching the ball
        if xy_dist <= 0.55:  # Define a catch range in the x-y plane
            reward = 1.0  # Catch success

        # # Reward for catching the ball and xy_dist minization
        # if abs(ball_pos[2] - net_pos[2]) <= 0.1 and xy_dist <= 0.55:
        #     reward = 1.0  # Catch success
        # elif xy_dist <= 0.55:
        #     reward = 0.5  # xy_dist minization

        # # Add a smoothness penalty based on joint angle changes
        # joint_smoothness_penalty = 0.1 * (abs(target_base_angle - curr_base_angle) + abs(target_shoulder_angle - curr_shoulder_angle))
        # reward -= joint_smoothness_penalty

        # Return updated observation, reward, done flag, and truncated flag
        obs = self._get_obs()
        done = self.step_counter >= self.max_ep_steps

        truncated = False  # We’re not using time-based or manual truncation yet
        return obs, reward, done, truncated, {}
    
    def _get_obs(self):
        # Get the ball's position
        ball_pos = pyB.getBasePositionAndOrientation(self.ball_id)[0]
        noisy_ball_x = ball_pos[0] + np.random.normal(0, self.ball_noise_std)
        noisy_ball_y = ball_pos[1] + np.random.normal(0, self.ball_noise_std)

        # Get the net's position
        net_pos = pyB.getLinkState(self.robot_id, self.net_joint_idx)[0]

        # Return the x and y coordinates of the ball and net (no joint positions or velocities)
        obs = np.array([noisy_ball_x, noisy_ball_y, net_pos[0], net_pos[1]], dtype=np.float32)    
        return obs

    def _get_pybullet(self):
        return pyB

    def close(self):
        pyB.disconnect()

    def draw_circle(self, radius, color):
        num_segments = 50
        for i in range(num_segments):
            theta1 = -np.pi/2 + (i / num_segments) * np.pi
            theta2 = -np.pi/2 + ((i + 1) / num_segments) * np.pi
            p1 = [radius * np.cos(theta1), radius * np.sin(theta1), 0.01]
            p2 = [radius * np.cos(theta2), radius * np.sin(theta2), 0.01]
            pyB.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=1.5)

def main():
    print("Testing Trainable Environment ...")
    env = LearmArmEnv(render=True)              # Create the simulation environment
    # obs = env.reset()                         # Reset and initialize the simulation
    delay_steps = 500

    for _ in range(env.max_ep_steps + delay_steps):     # Loop for max simulation steps + desired delay steps
        action = np.array([0.00, 1.00])                 # Move right + down smoothly
        obs, reward, done, _ = env.step(action)         # Apply the action and step the sim
        if done:
            break

    env.close()                                 # Disconnect from PyBullet when done


if __name__ == "__main__":
    main()