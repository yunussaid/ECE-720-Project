import gym
import numpy as np
import pybullet as pyB
import pybullet_data
from gym import spaces
import time


class LearmArmEnv(gym.Env):
    def __init__(self, render=False):
        super(LearmArmEnv, self).__init__()

        self.render_mode = render
        self.time_step = 1.0 / 240.0
        self.physics_client_id = None

        if self.render_mode:
            self.physics_client_id = pyB.connect(pyB.GUI)
        else:
            self.physics_client_id = pyB.connect(pyB.DIRECT)

        self.max_steps = 10000  # 240 = 1 second per episode
        self.step_counter = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)                # Actions: 2 joint angle velocities (continuous)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)     # Observations: 2 joint positions + 2 joint velocities

        self.base_joint_idx = None
        self.shoulder_joint_idx = None
        self.wrist_joint_idx = None

        self._setup_simulation()


    def _setup_simulation(self):
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


    def reset(self):
        self.step_counter = 0
        pyB.resetSimulation()
        self._setup_simulation()

        # Reset joints
        pyB.resetJointState(self.robot_id, self.base_joint_idx, 0)
        pyB.resetJointState(self.robot_id, self.shoulder_joint_idx, 0)
        pyB.resetJointState(self.robot_id, self.wrist_joint_idx, 0)

        return self._get_obs()

    def step(self, action):
        self.step_counter += 1

        # # Clip, scale and apply VELOCITY_CONTROL actions
        # action = np.clip(action, -1.0, 1.0)
        # target_vel_base = float(action[0])
        # target_vel_shoulder = float(action[1])
        # pyB.setJointMotorControl2(self.robot_id, self.base_joint_idx, pyB.VELOCITY_CONTROL, targetVelocity=target_vel_base)
        # pyB.setJointMotorControl2(self.robot_id, self.shoulder_joint_idx, pyB.VELOCITY_CONTROL, targetVelocity=target_vel_shoulder)

        # Get current joint positions
        base_pos = pyB.getJointState(self.robot_id, self.base_joint_idx)[0]
        shoulder_pos = pyB.getJointState(self.robot_id, self.shoulder_joint_idx)[0]
        
        # Get current joint positions and enforce joint limits
        delta_base, delta_shoulder = action
        new_base_pos = np.clip(base_pos + delta_base, -1.57, 1.57)
        new_shoulder_pos = np.clip(shoulder_pos + delta_shoulder, 0.0, 1.57)

        # Apply POSITION_CONTROL
        pyB.setJointMotorControl2(self.robot_id, self.base_joint_idx, pyB.POSITION_CONTROL, targetPosition=new_base_pos)
        pyB.setJointMotorControl2(self.robot_id, self.shoulder_joint_idx, pyB.POSITION_CONTROL, targetPosition=new_shoulder_pos)
        pyB.setJointMotorControl2(self.robot_id, self.wrist_joint_idx, pyB.POSITION_CONTROL, targetPosition=new_shoulder_pos)

        # Simulate one step
        pyB.stepSimulation()
        if self.render_mode:
            time.sleep(self.time_step)

        obs = self._get_obs()
        reward = 0  # TODO: Define task-specific reward
        done = self.step_counter >= self.max_steps

        return obs, reward, done, {}


    def _get_obs(self):
        joint_base = pyB.getJointState(self.robot_id, self.base_joint_idx)
        joint_shoulder = pyB.getJointState(self.robot_id, self.shoulder_joint_idx)

        pos = [joint_base[0], joint_shoulder[0]]
        vel = [joint_base[1], joint_shoulder[1]]
        return np.array(pos + vel, dtype=np.float32)

    def close(self):
        pyB.disconnect()



def main():
    print("Testing Trainable Environment ...")
    env = LearmArmEnv(render=True)  # Create the simulation environment
    # obs = env.reset()               # Reset and initialize the simulation

    for _ in range(2400):           # Loop for 240 simulation steps (1 second at 240Hz)
        # action = env.action_space.sample()     # Take a random action
        action = np.array([0.05, 0.05])  # Move right + down smoothly
        obs, reward, done, _ = env.step(action)  # Apply the action and step the sim
        if done:
            break

    env.close()  # Disconnect from PyBullet when done


if __name__ == "__main__":
    main()