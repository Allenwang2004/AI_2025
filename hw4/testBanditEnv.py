from BanditEnv import BanditEnv
import random

env = BanditEnv(10)
env.reset()

actions = []
rewards = []
for i in range(1000):
    action = random.randint(0, 9)
    reward = env.step(action)
    actions.append(action)
    rewards.append(reward)

action_history, reward_history = env.export_history()

for i in range(1000):
    assert actions[i] == action_history[i]
    assert rewards[i] == reward_history[i]