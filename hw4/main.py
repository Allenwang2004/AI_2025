import numpy as np
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent

def run_experiment(epsilon, runs=2000, steps=1000):
    avg_rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for _ in range(runs):
        env = BanditEnv(10)
        agent = Agent(10, epsilon)
        optimal_action = np.argmax(env.means)

        for step in range(steps):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

            avg_rewards[step] += reward
            if action == optimal_action:
                optimal_actions[step] += 1

    avg_rewards /= runs
    optimal_actions = (optimal_actions / runs) * 100
    return avg_rewards, optimal_actions

def run_non_stationary_experiment(epsilon, runs=2000, steps=10000):
    avg_rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for _ in range(runs):
        env = BanditEnv(10, stationary=False)
        agent = Agent(10, epsilon)
        for step in range(steps):
            optimal_action = np.argmax(env.means)
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

            avg_rewards[step] += reward
            if action == optimal_action:
                optimal_actions[step] += 1

    avg_rewards /= runs
    optimal_actions = (optimal_actions / runs) * 100
    return avg_rewards, optimal_actions

def run_constant_step_experiment(epsilon, alpha, runs=2000, steps=10000):
    avg_rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for _ in range(runs):
        env = BanditEnv(10, stationary=False)
        agent = Agent(10, epsilon, alpha)
        for step in range(steps):
            optimal_action = np.argmax(env.means)
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

            avg_rewards[step] += reward
            if action == optimal_action:
                optimal_actions[step] += 1

    avg_rewards /= runs
    optimal_actions = (optimal_actions / runs) * 100
    return avg_rewards, optimal_actions

def plot_results(results, steps, title_prefix, filename):
    plt.figure(figsize=(12, 5))

    # Plot Average Reward
    plt.subplot(1, 2, 1)
    for eps, (rewards, _) in results.items():
        plt.plot(steps, rewards, label=f'ε = {eps}')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'{title_prefix} - Average Reward')
    plt.legend()

    # Plot % Optimal Action
    plt.subplot(1, 2, 2)
    for eps, (_, optimal_actions) in results.items():
        plt.plot(steps, optimal_actions, label=f'ε = {eps}')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'{title_prefix} - % Optimal Action')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Stationary Experiment
    epsilons = [0, 0.01, 0.1]
    stationary_results = {}

    for eps in epsilons:
        rewards, optimal = run_experiment(eps)
        stationary_results[eps] = (rewards, optimal)

    plot_results(stationary_results, np.arange(1, 1001), 'Stationary', 'stationary_results.png')

    # Non-Stationary Experiment
    non_stationary_results = {}
    for eps in epsilons:
        rewards, optimal = run_non_stationary_experiment(eps)
        non_stationary_results[eps] = (rewards, optimal)

    plot_results(non_stationary_results, np.arange(1, 10001), 'Non-Stationary', 'non_stationary_results.png')

    # Constant Step Experiment
    constant_step_results = {}
    alphas = [0.1, 0.01, 0.001]
    for eps in epsilons:
        for alpha in alphas:
            rewards, optimal = run_constant_step_experiment(eps, alpha)
            constant_step_results[(eps, alpha)] = (rewards, optimal)
    plot_results(constant_step_results, np.arange(1, 10001), 'Constant Step', 'constant_step_results.png')