import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import DataCenterEnv

def evaluate_model(model, env, episodes=100):
    rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic policy

            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Average Reward on Unseen Data: {avg_reward}")
    return rewards


# Load the trained model
model = PPO.load("ppo_datacenter")

# Create validation environment
validation_env = DataCenterEnv("validate.xlsx")
# Run evaluation
validation_rewards = evaluate_model(model, validation_env, episodes=100)


plt.figure(figsize=(10, 5))
plt.plot(validation_rewards, label="Episode Reward")
plt.axhline(np.mean(validation_rewards), color='r', linestyle='--', label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Performance on Unseen Data")
plt.legend()
plt.show()