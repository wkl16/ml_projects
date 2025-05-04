from collections import defaultdict
import numpy as np

class Agent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.9, epsilon_greedy=0.9, epsilon_min=0.1, epsilon_decay=0.95):
        self.env = env
        self.actions = ["left", "right", "up", "down"]
        self.nA = len(self.actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def state_indices(self, state):
        return tuple(map(tuple, state))

    def choose_action(self, state):
        indices = self.state_indices(state)
        if np.random.uniform() < self.epsilon:
            action_idx = np.random.choice(self.nA)
        else:
            q_vals = self.q_table[indices]
            perm = np.random.permutation(self.nA)
            q_vals_shuffled = [q_vals[a] for a in perm]
            best_idx = np.argmax(q_vals_shuffled)
            action_idx = perm[best_idx]
        return action_idx

    def learn(self, transition):
        s, a, r, next_s, done = transition
        s_index = self.state_indices(s)
        next_s = self.state_indices(next_s)
        q_val = self.q_table[s_index][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[next_s])
        self.q_table[s_index][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
