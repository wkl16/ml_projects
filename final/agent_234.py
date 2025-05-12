# Task 2.3 - Building agent

from collections import defaultdict
import numpy as np

# Agent class is a modification of the "Sebastian Raschka. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022"
# It is modified to work with numpy array from environment written in section 2.2.
class Agent:
    """Agent for reinforcement learning.
    Parameters
    ------------
    env : 2d numpy array
    Environment that should be created from section 2.2
    learning_rate : float
    Learning rate (between 0.0 and 1.0)
    discount_factor : float
    Discount factor (between 0.0 and 1.0), value of future rewards
    epsilon_greedy : float
    Greedy policy (between 0.0 and 1.0)
    epsilon_min : float
    Lower bound for exploration (between 0.0 and 1.0)
    epsilon_decay : float
    Rate of reduction of exploration after learning (between 0.0 and 1.0)
    Attributes
    -----------
    actions : [String]
    Actions available to agent
    nA: int
    Number of actions
    gamma : float
    How much value in future rewards
    epsilon : float
    Greedy policy, probability of exploration
    q_table : defaultdict
    Dictionary that maps each state to an array of estimated action values (Q-values).
    """

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
        """Helper function that converts numpy array into indices that can be used by q_table's dictionary
        """
        return tuple(map(tuple, state))

    def choose_action(self, state):
        """Agent action decision based on random number falling within epsilon (exploration)
        or using the q-table (exploitation).
        Parameters
        ----------
        state : 2d numpy array
        current state of the environment
        Returns
        -------
        action_idx : int
        returns index of action taken
        """
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
        """Agent learning.
        Parameters
        ----------
        transition : a set of inputs as the following
            s : 2d numpy array
            current state of environment
            a : int
            action index
            r : int
            reward value
            next_s : 3d numpy array
            next state of environment
            done : int
            value to determine if finished
        Returns
        -------
        none
        """
        s, a, r, next_s, done = transition
        #get indices of current and next state
        s_index = self.state_indices(s)
        next_s = self.state_indices(next_s)
        #look up q table value of current state with desired action
        q_val = self.q_table[s_index][a]
        #if done return reward value
        if done:
            q_target = r
        #else update q_target
        else:
            q_target = r + self.gamma * np.max(self.q_table[next_s])
        #then update q table (at current state and action) and update epsilon.
        self.q_table[s_index][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        """Helper function that updates epsilon.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
