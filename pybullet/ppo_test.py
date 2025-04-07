from train_env import LearmArmEnv
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("ppo_learm_agent")
env = LearmArmEnv(render=True)
success_count = 0
num_trials = 20

for i in range(num_trials):
    obs, _ = env.reset()
    caught = False
    for j in range(env.max_ep_steps):
        action, _ = model.predict(obs, deterministic=True)
        if j % 100 == 0: # Print action every 100 steps (optional)
            print("action: ", [round(action[0], 2), round(action[1], 2)])
        
        # Take a step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Get the net and ball positions from the environment
        pyB = env._get_pybullet()  # Access the PyBullet client from the environment
        ball_pos = pyB.getBasePositionAndOrientation(env.ball_id)[0]
        net_pos = pyB.getLinkState(env.robot_id, env.net_joint_idx)[0]
        
        # Check if the ball is caught based on the condition (x-y distance + z distance)
        xy_dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(net_pos[:2]))
        if abs(ball_pos[2] - net_pos[2]) <= 0.1 and xy_dist <= 0.55:
            caught = True
            break

    ball_xy = tuple(round(coord, 1) for coord in ball_pos[:2])
    net_xy = tuple(round(coord, 1) for coord in net_pos[:2])
    print("ball_xy:", ball_xy, "| net_xy:", net_xy, "| dist:", round(xy_dist, 1))
    print(f"Trial {i+1}: {'Caught ✅' if caught else 'Missed ❌'}")
    success_count += int(caught)

print(f"\nPPO success rate: {success_count} / {num_trials}")
env.close()