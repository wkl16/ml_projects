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

def q_learn_stuff(agent, env, num_episodes=10000):
    history = []
    num_pos = 0
    num_neu = 0
    num_neg = 0
    pbar = pyprind.ProgBar(num_episodes, stream=sys.stdout)

    for episode in range(num_episodes):
        env.reset()
        state = env
        final_reward, time_t = 0.0, 0.0
        done = False

        while not done:
            action_idx = agent.choose_action(state)
            action = agent.actions[action_idx]
            next_state, reward = env.interact(action)
            done = reward == env.pos_reward
            agent._learn(Transition(state, action_idx, reward, next_state, done))
            state = next_state
            final_reward = reward
            time_t += 0.5
            if (time_t >= 20):
                break
        
        if final_reward == 0.0:
            num_neu = num_neu + 1
        elif final_reward < 0:
            num_neg = num_neg + 1
        else:
            num_pos = num_pos + 1

        
        history.append((time_t, final_reward))
        # Can uncomment to see these values, but now replacing with progbar
        # print(f'Episode {episode+1}: Reward {final_reward:.2f}, Time {time_t}')
        pbar.update()
    
    # Prints number of negative rewards, neutral rewards, and positive rewards respectively
    print(f'Num_Neg {num_neg} | Num_Neu {num_neu} | Num_Pos {num_pos}')
    return history

if __name__ == "__main__":
    env = CruiseEnv()
    agent = Agent_007(env)
    history = q_learn_stuff(agent, env)
