# Multi armed bandit
import numpy as np
import random
import statistics
import matplotlib.pyplot as plt


def epsilon_comparison_plot(title, x, y, x_label, y_label, save_path):

    for data, epsilon in y:
        plt.plot(x, data, label="epsilon: " + str(epsilon))

    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([15, 22])
    x_label = plt.xlabel(x_label)
    x_label.set_color('purple')
    y_label = plt.ylabel(y_label)
    y_label.set_color('green')
    plt.legend()
    plt.xscale('log')
    plt.savefig(save_path)
    plt.show()

def agent_comparison_plot(title, x, y, x_label, y_label, save_path):

    for data, agent in y:
        plt.plot(x, data, label="Agent: " + str(agent))

    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([15, 22])
    x_label = plt.xlabel(x_label)
    x_label.set_color('purple')
    y_label = plt.ylabel(y_label)
    y_label.set_color('green')
    plt.legend()
    plt.xscale('log')
    plt.savefig(save_path)
    plt.show()


class Bandit:

    def __init__(self, const_rew):
        self.const_rew = const_rew
        self.mean_payout = 0
        self.freq_mem = 0

    def pull(self):
        return np.random.randn() + self.const_rew * 10

"""
Simple agent uses epsilon-greedy policy to solve the multi-armed bandit problem.
"""
class Agent:

    def __init__(self, epsilon, decay_rate, bandits_available):
        self.action_space = bandits_available
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.bandits_available = bandits_available
        self.reward_mem = np.zeros([self.action_space]) # stores reward history
        self.freq_mem = np.zeros([self.action_space]) # stores number of time each bandit was pulled


    def act(self):
        if np.random.rand() < self.epsilon:
            self.epsilon *= self.decay_rate
            return random.randrange(self.action_space)
        else:
            self.epsilon *= self.decay_rate
            return np.argmax(self.reward_mem)


    def learn(self, action, last_payout):
        self.freq_mem[action] += 1
        self.reward_mem[action] = (1 - (1 / self.freq_mem[action])) * self.reward_mem[action] + \
                                  (1 / self.freq_mem[action] * last_payout)

"""
Optimistic agent uses greedy policy and begins with an inflated idea of what payouts each bandit will have then 
gradually lowers expectations.
"""
class OptimisticAgent:

    def __init__(self, decay_rate, bandits_available, optimism=30):
        self.action_space = bandits_available
        self.epsilon = None
        self.decay_rate = decay_rate
        self.bandits_available = bandits_available
        self.reward_mem = np.ones([self.action_space]) * optimism # stores reward history
        self.freq_mem = np.zeros([self.action_space]) # stores number of time each bandit was pulled


    def act(self):
        return np.argmax(self.reward_mem)


    def learn(self, action, last_payout):
        self.freq_mem[action] += 1
        self.reward_mem[action] = (1 - (1 / self.freq_mem[action])) * self.reward_mem[action] + \
                                  (1 / self.freq_mem[action] * last_payout)



"""
Upper confidence bound agent progresses towards greedy as upper bound gets smaller which happens
as all the bandits are tried more often.
"""
class UCBAgent:

    def __init__(self, decay_rate, bandits_available, optimism=30):
        self.action_space = bandits_available
        self.decay_rate = decay_rate
        self.bandits_available = bandits_available
        self.reward_mem = np.ones([self.action_space]) * optimism # stores reward history
        self.freq_mem = np.zeros([self.action_space]) # stores number of time each bandit was pulled


    def upper_conf_bound(self, b_mean, b_pulls):
        if b_pulls < 1:
            return float('inf')

        return b_mean + np.sqrt(2 * np.log(self.bandits_available) / b_pulls)


    def act(self):
        return np.argmax(
            [self.upper_conf_bound(self.reward_mem[i],
                                   self.freq_mem[i]) for i in range(self.bandits_available)])


    def learn(self, action, last_payout):
        self.freq_mem[action] += 1
        self.reward_mem[action] = (1 - (1 / self.freq_mem[action])) * self.reward_mem[action] + \
                                  (1 / self.freq_mem[action] * last_payout)




def create_bandits(nb_bandits):
    return [Bandit(i) for i in range(nb_bandits)]


def run_trials(nb_episodes, nb_bandits, epsilon=None, agent_type='standard', optimism=30):

    bandits = create_bandits(nb_bandits)


    if agent_type == 'optimistic':
        agent = OptimisticAgent(0.99, nb_bandits, optimism)

    elif agent_type.upper() == 'UCB':
        agent = UCBAgent(0.99, nb_bandits, optimism)

    else:
        agent = Agent(epsilon, 0.99, nb_bandits)

    total_reward = 0
    rewards = []
    snapshot_means = []
    snapshot_freq = 10
    cum_means = []

    for e in range(1, nb_episodes + 1):
        action = agent.act()
        reward = bandits[action].pull()

        total_reward += reward
        rewards.append(reward)
        agent.learn(action, reward)

        if e % snapshot_freq == 0:
            snapshot_means.append(statistics.mean(rewards))
            cum_means.append(statistics.mean(snapshot_means))
            rewards = []

            print("Trial {} of {} -- score: {}".format(e, str(nb_episodes + 1), reward))

    print("Total reward: {}\n".format(total_reward))

    return snapshot_freq, snapshot_means, cum_means, nb_episodes

"""
AVAILABLE AGENTS
    standard (epsilon-greedy)   |   optimistic  |   ucb (Upper Confidence Bound)
"""

"""EPSILON COMPARISON"""

epsilons = [0.3, 0.5, 0.9]
y = []

for epsilon in epsilons:
    snapshot_freq, snapshot_means, cum_means, nb_episodes = run_trials(10000, 3, epsilon, 'standard', 30)

    x = range(snapshot_freq, nb_episodes + 1, snapshot_freq)
    y.append((cum_means, epsilon))

epsilon_comparison_plot("Multi-armed bandit trials comparing epsilons using 'Epsilon-Greedy'", x, y,
                        "Trials", "Mean Scores",
                "multi armed bandit/Multi Armed Bandit - Epsilon Greedy.png")


"""AGENT COMPARISON"""

agents = ['standard', 'optimistic', 'UCB']
y = []

for agent in agents:
    snapshot_freq, snapshot_means, cum_means, nb_episodes = run_trials(10000, 3, 0.5, agent, 30)

    x = range(snapshot_freq, nb_episodes + 1, snapshot_freq)
    y.append((cum_means, agent))

agent_comparison_plot("Results from multi-armed bandit trials comparing agents",
                        x, y, "Trials", "Mean Scores",
                "multi armed bandit/Multi Armed Bandit - Agent-comparison.png")