from Agent import Agent

k = 10
epsilon = 0.1
agent = Agent(k, epsilon)
action = agent.select_action()
reward  = 0
agent.update_q(action, reward)