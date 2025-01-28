from env import DataCenterEnv
from ppo import PPO
import optuna

# Optuna optimization function
def objective(trial):
   # Hyperparameter search space
   lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
   discount_factor = trial.suggest_uniform('discount_factor', 0.8, 0.99)
   num_updates_per_iteration = trial.suggest_int('num_updates_per_iteration', 5, 20)

   # Initialize environment and PPO agent
   env = DataCenterEnv()
   ppo_agent = PPO(env, lr, discount_factor, num_updates_per_iteration)

   # Define total timesteps for training
   total_timesteps = 10000

   # Train the PPO agent
   ppo_agent.learn(total_timesteps)

   # Return the negative of the reward (because Optuna minimizes the objective)
   # Here, you can replace with a specific evaluation metric (e.g., average reward)
   return -ppo_agent.evaluate_performance()

# Run the optimization with Optuna
study = optuna.create_study(direction="minimize")  # Minimize negative reward to maximize reward
study.optimize(objective, n_trials=50)  # Run 50 trials

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)