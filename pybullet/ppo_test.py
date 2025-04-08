import os
import csv
from train_env import LearmArmEnv
from stable_baselines3 import PPO
import numpy as np

num_trials = 1000
visualize = True if num_trials <= 20 else False
env = LearmArmEnv(render=visualize)
model = PPO.load("ppo_learm_agent")
success_count = 0
results = []

for i in range(num_trials):
    obs, _ = env.reset()
    caught = False
    for j in range(env.max_ep_steps):
        action, _ = model.predict(obs, deterministic=True)
        
        # Print action every 100 steps (optional for debugging)
        if num_trials <= 20 and j % 100 == 0:
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

    # Append trial result for plotting and print trial result
    success_count += int(caught)
    results.append([i+1, success_count])
    print(f"Trial {i+1}: {'Caught ✅' if caught else 'Missed ❌'}")

print(f"\nPPO success rate: {success_count} / {num_trials}")

# Save results to a CSV file in the "results" folder
if not os.path.exists("results"):
    os.makedirs("results")  # Create the folder if it doesn't exist

# Save results with the model's learning rate information in the filename
filename = f'results/PPO_0003_results.csv'  # Use '0003' to match the learning rate in the filename

# Write the results to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Trial', 'Cumulative Success Count'])  # Header row
    writer.writerows(results)  # Write the results for each trial

env.close()