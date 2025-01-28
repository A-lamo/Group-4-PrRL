from env import DataCenterEnv
from ppo import PPO
import numpy as np

path_to_dataset = "../train.xlsx"
env = DataCenterEnv(path_to_dataset)
model = PPO(env)
print("model created!")
print("training in progress")
model.learn(300000)
print("training complete :)")


print("How well does it perform on the training data?")
aggregate_reward = 0
terminated = False
state = env.reset()
while not terminated:
    # agent is your own imported agent class
    action, _ = model.policy(state)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = env.step(action)
    state = next_state
    aggregate_reward += reward
    # print("Action:", action)
    # print("Next state:", next_state)
    # print("Reward:", reward)

print('Total reward on training data:', aggregate_reward)

print("starting testing phase")
env = DataCenterEnv("../validate.xlsx")
aggregate_reward = 0
terminated = False
state = env.observation()
while not terminated:
    # agent is your own imported agent class
    action, _ = model.policy(state)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = env.step(action)
    state = next_state
    aggregate_reward += reward
    # print("Action:", action)
    # print("Next state:", next_state)
    # print("Reward:", reward)

print('Total reward on the validation dataset:', aggregate_reward)