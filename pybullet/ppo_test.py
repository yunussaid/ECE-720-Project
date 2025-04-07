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
        if j % 100 == 0:
            print("action: ", [round(action[0], 2), round(action[1], 2)])
        # action[1] = -1 * action[1]
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward == 1.0:
            caught = True
            break
    print(f"Trial {i+1}: {'Caught ✅' if caught else 'Missed ❌'}")
    success_count += int(caught)

print(f"\nPPO success rate: {success_count} / {num_trials}")
env.close()