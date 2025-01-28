import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from matplotlib import cm
import pandas as pd
import seaborn as sns
from env import DataCenterEnv
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure





log_dir = "./ppo_logs/"
env = DummyVecEnv([lambda: DataCenterEnv("train.xlsx")])

model = PPO(
    "MlpPolicy",  # Use "CustomLSTMPolicy" if using LSTM
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    device="cpu"
)

model.learn(total_timesteps=300000, )
# model.save("ppo_datacenter")


"""
# Evaluate the model
env = DataCenterEnv("validate.xlsx")
obs = env.reset()
done = False
episode_reward = 0
model = PPO.load("ppo_datacenter")

max_episodes = 10
reward_logger = []

for episode in range(max_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    print('In episode:', episode)
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    
    reward_logger.append(episode_reward)  # Append episode reward

    if episode % 2 == 0:
        print(f"Episode {episode}: Reward = {episode_reward}")

# Plot the reward curve
plt.figure(figsize=(10, 5))
plt.plot(reward_logger, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training Performance")
plt.legend()
plt.show()

"""