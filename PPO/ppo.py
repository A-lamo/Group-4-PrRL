from network import FeedForwardNN
from env import DataCenterEnv
import torch
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn
import numpy as np


class PPO:
   def __init__(self, env):
      self.env = env
      self.obs_dim = env.observation().shape[0]
      self.act_dim = 3  # Discrete actions: 0, 1, -1

      # Set device (GPU if available, else CPU)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self._init_hyperparams()

      # Initialize networks and move them to the device
      self.actor = FeedForwardNN(self.obs_dim, self.act_dim).to(self.device)
      self.critic = FeedForwardNN(self.obs_dim, 1).to(self.device)

      self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
      self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

   def _init_hyperparams(self):
      self.timesteps_per_batch = 1000
      self.max_timesteps_per_episode = 100
      self.discount_factor = 0.9
      self.num_updates_per_iteration = 5
      self.lr = 5e-3
      self.clip = 0.2

   def learn(self, total_timesteps):
      t_so_far = 0

      while t_so_far < total_timesteps:
         if t_so_far % 100 == 0:
            print("Step t : ", t_so_far)
         batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths = self.rollout()
         t_so_far += np.sum(batch_lengths)
         V, _ = self.evaluate(batch_obs, batch_acts)

         A_k = batch_rtgs - V.detach()

         A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

         for i in range(self.num_updates_per_iteration):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1+self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()


   def policy(self, obs):
      obs = torch.tensor(obs, dtype=torch.float, device=self.device)  # Move observation to device
      logits = self.actor(obs)
      dist = Categorical(logits=logits)  # Using Categorical distribution for discrete actions

      action = dist.sample()
      log_prob = dist.log_prob(action)
      return action.cpu().numpy(), log_prob.cpu()  # Move action and log_prob back to CPU

   def compute_rtgs(self, batch_rewards):
      batch_rtgs = []

      for ep_rewards in reversed(batch_rewards):
         discounted_reward = 0

         for reward in reversed(ep_rewards):
            discounted_reward = reward + self.discount_factor * discounted_reward
            batch_rtgs.insert(0, discounted_reward)
         
      batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)  # Move to device
      return batch_rtgs
   
   def evaluate(self, batch_obs, batch_acts):
      V = self.critic(batch_obs).squeeze()
      
      logits = self.actor(batch_obs)
      dist = Categorical(logits=logits)
      log_probs = dist.log_prob(batch_acts)

      return V, log_probs

   def rollout(self):
      batch_obs = []
      batch_acts = []
      batch_log_probs = []
      batch_rewards = []
      batch_reward_to_gos = []
      batch_lengths = []

      t = 0

      while t < self.timesteps_per_batch:
         ep_rewards = []
         obs = self.env.reset()
         done = False

         for ep_t in range(self.max_timesteps_per_episode):
            t += 1

            batch_obs.append(obs)
            action, log_prob = self.policy(obs)

            obs, reward, done = self.env.step(action)

            ep_rewards.append(reward)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)

            if done:
               break
         
         batch_lengths.append(ep_t + 1)
         batch_rewards.append(ep_rewards)

      # Convert lists to tensors and move them to the device
      batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
      batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
      batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)

      batch_reward_to_gos = self.compute_rtgs(batch_rewards)

      return batch_obs, batch_acts, batch_log_probs, batch_reward_to_gos, batch_lengths


   def evaluate_performance(self):
      env = DataCenterEnv("../validate.xlsx")
      aggregate_reward = 0
      terminated = False
      state = env.observation()
      while not terminated:
         # agent is your own imported agent class
         action, _ = self.policy(state)
         # next_state is given as: [storage_level, price, hour, day]
         next_state, reward, terminated = env.step(action)
         state = next_state
         aggregate_reward += reward
         # print("Action:", action)
         # print("Next state:", next_state)
         # print("Reward:", reward)

      print('Total reward on the validation dataset:', aggregate_reward)