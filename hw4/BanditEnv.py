import numpy as np

class BanditEnv:
    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary
        self.reset()

    def reset(self):
        self.means = np.random.normal(0, 1, self.k)
        self.action_history = []
        self.reward_history = []

    def step(self, action):
        if not self.stationary:
            self.means += np.random.normal(0, 0.01, self.k)

        reward = np.random.normal(self.means[action], 1)
        self.action_history.append(action)
        self.reward_history.append(reward)
        return reward

    def export_history(self):
        return self.action_history, self.reward_history