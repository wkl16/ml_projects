import pyprind
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from EnvAgt import ParkinEnv
from EnvAgt import Agent_007

np.random.seed(42)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def q_learn_stuff(agent, num_episodes=50000):
    history = []
    num_pos = 0
    num_neu = 0
    num_neg = 0
    pbar = pyprind.ProgBar(num_episodes, stream=sys.stdout)

    for episode in range(num_episodes):
        agent.get_env().reset()
        state = agent.get_env()
        final_reward, time_t = 0.0, 0.0
        done = False
        state_table = []
        state_table.append([state.get_state(), time_t])

        while not done:
            action_idx = agent.choose_action(state)
            action = agent.actions[action_idx]
            next_state, reward = agent.env.interact(action)
            time_t += 0.5
            done = (time_t >= 20) # or (reward <= -100)
            agent._learn(Transition(state, action_idx, reward, next_state, done))
            state = next_state
            final_reward = reward
            state_table.append([state, time_t])
        
        if final_reward <= -100:
            num_neg = num_neg + 1
        elif final_reward < 0:
            num_neu = num_neu + 1
        else:
            num_pos = num_pos + 1

        
        history.append(((time_t, final_reward), state_table))
        # Can uncomment to see these values, but now replacing with progbar
        # print(f'Episode {episode+1}: Reward {final_reward:.2f}, Time {time_t}')
        pbar.update()
        # print(f'Num_Neg {num_neg} | Num_Neu {num_neu} | Num_Pos {num_pos}')
        # print(state_table)
    print(history[-1][1])
    # Prints number of negative rewards, neutral rewards, and positive rewards respectively
    print(f'Num_Neg {num_neg} | Num_Neu {num_neu} | Num_Pos {num_pos}')
    
    return history

if __name__ == "__main__":
    env = ParkinEnv(discretize_val=101, s0=(1,-1))
    agent = Agent_007(env)
    history = q_learn_stuff(agent)
