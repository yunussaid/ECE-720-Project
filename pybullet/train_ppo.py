from train_env import LearmArmEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = DummyVecEnv([lambda: LearmArmEnv(render=False)])

# Optional sanity check
check_env(LearmArmEnv(render=False))

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_log", device="auto")
model.learn(total_timesteps=150_000)  # adjust based on available time

# Save model
model.save("ppo_learm_agent")