from env import DataCenterEnv
import numpy as np
import argparse
from heuristic_agent import HeuristicAgent
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='validate.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path



all_rewards = []
long_rewards = []
long_actions = []
for i in range(10):
    environment = DataCenterEnv(path_to_dataset)

    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    agent = HeuristicAgent()
    rewards = []
    actions = []
    while not terminated:
        # action = agent.act(state)
        action = np.random.uniform(-1, 1)
        actions.append(action)
        next_state, reward, terminated = environment.step(action)
        state = next_state
        aggregate_reward += reward
        rewards.append(reward)
    #     print("Action:", action)
    #     print("Next state:", next_state)
    #     print("Reward:", reward)

    print('Total reward:', aggregate_reward)
    long_rewards.append(rewards)
    long_actions.append(actions)
    all_rewards.append(aggregate_reward)

long_rewards = np.array(long_rewards)
long_actions = np.array(long_actions)

# Calculate the average reward per time step
avrg_long_rewards = np.mean(long_rewards, axis=0)
# Calculate the standard deviation for rewards
std_long_rewards = np.std(long_rewards, axis=0)

# Calculate the average action per time step
avrg_long_actions = np.mean(long_actions, axis=0)
# Calculate the standard deviation for actions
std_long_actions = np.std(long_actions, axis=0)

# Plotting rewards with shaded std region
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(avrg_long_rewards, label='Average Reward', linewidth=2)  # Increased line thickness
plt.fill_between(range(len(avrg_long_rewards)), avrg_long_rewards - std_long_rewards, avrg_long_rewards + std_long_rewards, color='gray', alpha=0.5, label='Reward Std Dev')
plt.xlabel('Time Step', fontsize=14, fontweight='bold')  # Increased font size and bold
plt.ylabel('Reward', fontsize=14, fontweight='bold')  # Increased font size and bold
plt.title('Average Rewards with Std Dev Shaded Region', fontsize=16, fontweight='bold')  # Increased font size and bold
plt.legend(fontsize=12, frameon=False)  # Increased legend font size

# Plotting actions with shaded std region
plt.subplot(2, 1, 2)
plt.plot(avrg_long_actions, label='Average Action', color='orange', linewidth=2)  # Increased line thickness
plt.fill_between(range(len(avrg_long_actions)), avrg_long_actions - std_long_actions, avrg_long_actions + std_long_actions, color='gray', alpha=0.5, label='Action Std Dev')
plt.xlabel('Time Step', fontsize=14, fontweight='bold')  # Increased font size and bold
plt.ylabel('Action', fontsize=14, fontweight='bold')  # Increased font size and bold
plt.title('Average Actions with Std Dev Shaded Region', fontsize=16, fontweight='bold')  # Increased font size and bold
plt.legend(fontsize=12, frameon=False)  # Increased legend font size

# Display the plots
plt.tight_layout()
plt.show()

# Print the overall average reward
print('Average reward:', np.mean(all_rewards))