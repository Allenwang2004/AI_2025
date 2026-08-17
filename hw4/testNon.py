from Agent import Agent
from BanditEnv import BanditEnv

k = 10
epsilon = 0.1

# Original stationary interface should work as well
#  # env = BanditEnv(k)

env = BanditEnv(k, stationary=False)
agent = Agent(k, epsilon)