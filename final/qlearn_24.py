# Task 2.4 - Q-learning process

import pyprind
from environment_22 import Environment
from agent_234 import Agent
from collections import namedtuple
from map_abstraction_21 import Map_Abstraction
import matplotlib.pyplot as plt
import numpy as np
from test2 import plot_learning_history

np.random.seed(1)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# objective: implement Q-learning function which repeatedly performs agent-environment interactions for a specific number of episdoes which are length-bound. 
# use progress bar during training - https://github.com/rasbt/pyprind/blob/master/README.md
# function should have following parameters: learning rate, discount factor, number of episodes, max length of episodes
# may tune epsilon and use as probability of exploration
def qlearn (lr, df, num_episodes, max_ep_len, env):
    history = []
    agent = Agent(env, learning_rate=lr, discount_factor=df)
    
    pbar = pyprind.ProgBar(num_episodes, title='Training Progress')
    
    for episode in range(num_episodes):
        state = env.reset()
        final_reward, n_moves = 0.0, 0
        done = False

        for s in range(max_ep_len):
            action_idx = agent.choose_action(state)
            action = agent.actions[action_idx]
            next_state, reward = env.interaction(state.copy(), action)
            done = reward == env.goal_reward  # goal reached
            agent.learn(Transition(state, action_idx, reward, next_state, done))
            state = next_state.copy()
            n_moves += 1
            final_reward = reward  # store final reward for logging

            if done:
                break
    
        history.append((n_moves, final_reward))
        pbar.update()

        #print(f'Episode {episode}: Reward {final_reward:.2f}, #Moves {n_moves}')
    
    #print(pbar)
    return history

if __name__ == '__main__':
    ma = Map_Abstraction()
    #np_img_array = ma.abstract("./maps/map1.bmp", 20, 20)
    np_img_array = ma.abstract("./maps/map2.bmp", 40, 40)
    # CHANGE MAP HERE
    env = Environment(np_img_array, 0, 0, 38, 38)
    history = qlearn(0.01, 0.9, 300, 30000, env)
    plot_learning_history(history)