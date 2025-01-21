import numpy as np

class HeuristicAgent:
    def __init__(self):
        self.storage_level = 0
        self.price = 0
        self.hour = 0
        self.day = 0

    def act(self, state):
        self.storage_level = state[0]
        self.price = state[1]
        self.hour = state[2]
        self.day = state[3]

        if 0 < self.hour <= 8:
            return 1.0
        else:
            return np.random.uniform(-1, 1)
        
