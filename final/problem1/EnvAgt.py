from scipy.integrate import odeint
from collections import defaultdict
import numpy as np

### vv Section for globals vv ###

np.random.seed(42)
control_inputs = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0, 0.0001, 0.001, 0.01, 0.1, 1, 5])

### ^^ Section for Globals ^^ ###

### vv Necessary for calculating next state vv ###

def model(s, t, u):
    dsdt = [s[1],u]
    return dsdt

delta = 0.1
step = np.linspace(0, delta)

'''Example for next state'''
# y = odeint(model, s0, step, args=(u_i))

### ^^ Necessary for calculating next state ^^ ###

# Environment Class
class ParkinEnv:
    # Initialize everything
    def __init__(self, discretize_val=51, neg_reward=-100, neutral_reward=-2, pos_reward=50, s0=(1,0)):
        self.discretize_val = discretize_val
        self.neg_reward = neg_reward
        self.neutral_reward = neutral_reward
        self.pos_reward = pos_reward
        self.s0 = s0
        self.s0_copy = s0

        # Create discretized state space from continuous space
        self.X, self.V, self.x_sp, self.v_sp = self.state_space_discretize(self.discretize_val)

    # To discretize for Task 1.5, initial tests will be run
    # using default n = 21
    def state_space_discretize(self, n=21):
        X = np.linspace(-5, 5, n)
        V = np.linspace(-5, 5, n)
        x_space = X[1] - X[0]
        v_space = V[1] - V[0]
        return X, V, x_space, v_space
    
    # Map continuous results from odeint to discrete space
    # return indices of position and velocity
    def map_indexes(self, state):
        pos_idx = int(round((state[0] - self.X[0]) / self.x_sp))
        vel_idx = int(round((state[1] - self.V[0]) / self.v_sp))

        return pos_idx, vel_idx
    
    # Interact with environment and give positive/negative reward
    def interact(self, control):
        # Check if already at goal first
        if self.s0 == (0,0):
            return self.s0, self.pos_reward

        # initialize neutral reward
        reward = self.neutral_reward

        # control var for odeint
        u = control

        # calculate ode
        ode_result = odeint(model, self.s0, step, args=(u,))

        # get mapped result, this may be the problem
        pos_idx, vel_idx = self.map_indexes(ode_result[-1])
        
        # initialize out of bounds param
        out_of_bounds = False
        
        # Do bound checking so that next state does not crash program
        if pos_idx >= self.discretize_val or pos_idx < 0:
            out_of_bounds = True
            if pos_idx >= self.discretize_val:
                pos_idx = self.discretize_val - 1
            else:
                pos_idx = 0
        if vel_idx >= self.discretize_val or vel_idx < 0:
            out_of_bounds = True
            if vel_idx >= self.discretize_val:
                vel_idx = self.discretize_val - 1
            else:
                vel_idx = 0

        # Penalize if oob
        if out_of_bounds:
            reward = self.neg_reward
        # Reward if got to pos 0 vel 0
        else:
            if (pos_idx == (self.discretize_val // 2) and vel_idx == (self.discretize_val // 2)):
                reward = self.pos_reward
        
        # get new state based off of ode
        new_state = (self.X[pos_idx], self.V[vel_idx])
        
        # update state
        self.s0 = new_state
        return new_state, reward 
    
    # Reset s0
    def reset(self):
        self.s0 = self.s0_copy
        return self.s0

# Agent class
class Agent_007():
    # Initialize Everything
    def __init__(self, env, learning_rate=0.01, discount_factor=0.9, epsilon_greedy=0.9, epsilon_min=0.001, epsilon_decay=0.95):
        self.env       = env
        self.actions   = control_inputs
        self.nA        = len(self.actions)
        self.lr        = learning_rate
        self.gamma     = discount_factor
        self.eps       = epsilon_greedy
        self.eps_min   = epsilon_min
        self.eps_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

    # Choose a random action on self.actions
    def choose_action(self, state):
        # 
        if np.random.uniform() < self.eps:
            action = np.random.choice(self.nA)
        else:
            q_vals = self.q_table[state]
            perm_actions = np.random.permutation(self.nA)
            q_vals = [q_vals[a] for a in perm_actions]
            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]
        return action

    # Do learning, gotten from textbook
    def _learn(self, transition):
        s, a, r, next_s, done = transition
        q_val = self.q_table[s][a]
        
        if done:
            q_target = r
        else:
            q_target = r + self.gamma*np.max(self.q_table[next_s])
            
        self.q_table[s][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()
    
    # Adjust stuff, also gotten from textbook
    def _adjust_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
