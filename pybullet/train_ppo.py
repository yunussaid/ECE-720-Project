from train_env import LearmArmEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = DummyVecEnv([lambda: LearmArmEnv(render=False)])

# Optional sanity check
check_env(LearmArmEnv(render=False))

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_log", device="auto",
    # ent_coef=0.01,  # Entropy coefficient to promote exploration
    learning_rate=0.00003,  # Fine-tune learning rate
    # batch_size=1024,  # Adjust batch size for better exploration
)
model.learn(total_timesteps=150_000, tb_log_name="PPO_00003_1n")

# Save model
# model.save("ppo_agent")
model.save("./ppo_controllers/ppo_00003_1n")