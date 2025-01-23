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
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import seaborn as sns
from env import DataCenterEnv
import optuna

# Set the seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class RNNQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNNQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected output layer
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)


    def forward(self, x):
        batch_size = x.size(0)  # Ensure batch size is inferred dynamically
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]  # Take the last output in the sequence
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out


class ExperienceReplay:
    def __init__(self, env, buffer_size, min_replay_size=1000, seed=123):
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-9000000.0], maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print('Filling experience replay buffer with random transitions...')
                
        obs = self.env.reset() 
        for _ in range(self.min_replay_size):
            if 0 < obs[2] < 7:
                action = 1
            else:
                action = np.random.choice([0, 1, 2])  # Fix action sampling
            new_obs, reward, done = env.step(action)  # Fix step output

            transition = (obs, action, reward, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs = env.reset()

        print('Initialization with random transitions is done!')

    def sample(self, batch_size):
        transitions = random.sample(self.replay_buffer, batch_size)

        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t

    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)

    def add_reward(self, reward):
        
        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''
        
        self.reward_buffer.append(reward)
        

class vanilla_DQNAgent:
    def __init__(self, path_to_dataset, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed=123):
        self.env = DataCenterEnv(path_to_dataset)
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size

        self.replay_memory = ExperienceReplay(self.env, self.buffer_size, seed=seed)
        self.online_network = RNNQNetwork(4, 64, 3, num_layers=2).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), self.learning_rate)  # Define optimizer


    def choose_action(self, step, observation, greedy=False):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        if random.random() <= epsilon and not greedy:
            return np.random.choice([0, 1, 2]), epsilon

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # Fix shape for LSTM

        q_values = self.online_network(obs_t)
        action = torch.argmax(q_values, dim=1).item()
        return action, epsilon

    def learn(self, batch_size):
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        new_observations_t = new_observations_t.unsqueeze(1)  # Ensure correct shape for LSTM
        target_q_values = self.online_network(new_observations_t)  
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

        observations_t = observations_t.unsqueeze(1)  # Ensure correct shape for LSTM
        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(q_values, dim=1, index=actions_t)

        loss = F.smooth_l1_loss(action_q_values, targets.detach())

        self.optimizer.zero_grad()  # Use the optimizer defined in the agent
        loss.backward()
        self.optimizer.step()


def training_loop(path_to_dataset, agent, max_episodes, target_ = False, seed=42):
    
    '''
    Params:
    env = name of the environment that the agent needs to play
    agent= which agent is used to train
    max_episodes = maximum number of games played
    target = boolean variable indicating if a target network is used (this will be clear later)
    seed = seed for random number generator for reproducibility
    
    Returns:
    average_reward_list = a list of averaged rewards over 100 episodes of playing the game
    '''
    env = DataCenterEnv(path_to_dataset)
    obs = env.reset()
    average_reward_list = [-8000000.0]
    episode_reward = 0.0
    
    for step in range(max_episodes):
        
        action, epsilon = agent.choose_action(step, obs)
       
        new_obs, reward, terminated = env.step(action)
        done = terminated         
        transition = (obs, action, reward, done, new_obs)
        agent.replay_memory.add_data(transition)
        obs = new_obs
    
        episode_reward += reward
    
        if done:
        
            obs = env.reset()
            agent.replay_memory.add_reward(episode_reward)
            episode_reward = 0.0

        #Learn

        agent.learn(batch_size)

        #Calculate after each 100 episodes an average that will be added to the list
                
        if (step+1) % 100 == 0:
            average_reward_list.append(np.mean(agent.replay_memory.reward_buffer))
        
        # #Update target network, do not bother about it now!
        if target_:
            
            #Set the target_update_frequency
            target_update_frequency = 300
            if step % target_update_frequency == 0:
                dagent.update_target_network()

    
        #Print some output
        if (step+1) % 10000 == 0:
            print(20*'--')
            print('Step', step)
            print('Epsilon', epsilon)
            print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
            print()

    return average_reward_list


# vanilla_agent = vanilla_DQNAgent(path_to_dataset, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)
# average_rewards_vanilla_dqn = training_loop(path_to_dataset, vanilla_agent, max_episodes)

class DDQNAgent:
    
    def __init__(self, path_to_dataset, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
        '''
        Params:
        env = name of the environment that the agent needs to play
        device = set up to run CUDA operations
        epsilon_decay = Decay period until epsilon start -> epsilon end
        epsilon_start = starting value for the epsilon value
        epsilon_end = ending value for the epsilon value
        discount_rate = discount rate for future rewards
        lr = learning rate
        buffer_size = max number of transitions that the experience replay buffer can store
        seed = seed for random number generator for reproducibility
        '''
        self.env_name = path_to_dataset
        self.env = DataCenterEnv(path_to_dataset)
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        self.replay_memory = ExperienceReplay(self.env, self.buffer_size, seed = seed)
        self.online_network = RNNQNetwork(4, 64, 3, num_layers=2).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), self.learning_rate)  # Define optimizer

        
        '''
        ToDo: Add here a target network and set the parameter values to the ones of the online network!
        Hint: Use the method 'load_state_dict'!
        '''
        
        #Solution:
        self.target_network = RNNQNetwork(4, 64, 3, num_layers=2).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
    def choose_action(self, step, observation, greedy=False):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        if random.random() <= epsilon and not greedy:
            return np.random.choice([0, 1, 2]), epsilon

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # Fix shape for LSTM

        q_values = self.online_network(obs_t)
        action = torch.argmax(q_values, dim=1).item()
        return action, epsilon
    
        
    def learn(self, batch_size):
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        new_observations_t = new_observations_t.unsqueeze(1)  # Ensure correct shape for LSTM
        target_q_values = self.online_network(new_observations_t)  
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

        observations_t = observations_t.unsqueeze(1)  # Ensure correct shape for LSTM
        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(q_values, dim=1, index=actions_t)

        loss = F.smooth_l1_loss(action_q_values, targets.detach())

        self.optimizer.zero_grad()  # Use the optimizer defined in the agent
        loss.backward()
        self.optimizer.step()

        
    def update_target_network(self):
        
        '''
        ToDO: 
        Complete the method which updates the target network with the parameters of the online network
        Hint: use the load_state_dict method!
        '''
    
        #Solution:
        
        self.target_network.load_state_dict(self.online_network.state_dict())
    

#Discount rate
discount_rate = 0.99
#That is the sample that we consider to update our algorithm
batch_size = 128
#Maximum number of transitions that we store in the buffer
buffer_size = 50000
#Minimum number of random transitions stored in the replay buffer
min_replay_size = 1000
#Starting value of epsilon
epsilon_start = 1.0
#End value (lowest value) of epsilon
epsilon_end = 0.05
#Decay period until epsilon start -> epsilon end
epsilon_decay = 200000

max_episodes = 300000

#Learning_rate
lr = 5e-2
#Path to the dataset
path_to_dataset = "train.xlsx"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dagent = DDQNAgent(path_to_dataset, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)
average_rewards_ddqn = training_loop(path_to_dataset, dagent, max_episodes, target_ = True) 