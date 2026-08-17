import numpy as np

class Agent:
    def __init__(self, k, epsilon, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)

    def update_q(self, action, reward):
        self.N[action] += 1
        if self.alpha is not None:
            # Constant step-size
            self.Q[action] += self.alpha * (reward - self.Q[action])
        else:
            # Sample-Average
            self.Q[action] += (reward - self.Q[action]) / self.N[action]