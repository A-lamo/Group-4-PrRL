from env import DataCenterEnv
import numpy as np
import argparse
import matplotlib.pyplot as plt


args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)

aggregate_reward = 0
terminated = False
state = environment.observation()

avg = []

for i in range(24):
    avg.append(environment.price_values[:,i])

avg = np.array(avg)
avg = np.mean(avg, axis=0)
print(avg)
plt.plot(avg)
plt.xlabel('Day from dataset')
plt.ylabel('Price')
plt.title('Average price per day')
plt.show()

avg = []

for i in range(1096):
    avg.append(environment.price_values[i,:])
    
avg = np.array(avg)
avg = np.mean(avg, axis=0)
print(avg)
plt.plot(avg)
plt.axhline(y=np.mean(avg), color='red', linestyle='dotted', linewidth=2, label=f'Avg: {np.mean(avg):.2f}')
plt.xlabel('Hour of day')
plt.ylabel('Price')
plt.title('Average price per hour')
plt.show()