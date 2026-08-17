from Agent import Agent

k = 10
epsilon = 0.1

# Original stationary interface should work as well
# agent = Agent(k, epsilon)

# Construct with None should behave the same as the original
# agent = Agent(k, epsilon, None)

agent = Agent(k, epsilon, 0.1)
action = agent.select_action()
reward = 0
agent.update_q(action, reward)