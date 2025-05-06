import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from EnvAgt import CruiseEnv
from EnvAgt import Agent_007

np.random.seed(42)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Q Learning stuff as defined in textbook, but slightly modified for cruise control problem
def q_learn_stuff(agent, env, num_episodes=5000):
    history = []
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
            print(next_state)
            final_reward = reward
            time_t += 0.5
            if (time_t >= 20):
                break
        history.append((time_t, final_reward))
        print(f'Episode {episode+1}: Reward {final_reward:.2f}, Time {time_t}')
    return history

if __name__ == "__main__":
    env = CruiseEnv()
    agent = Agent_007(env)
    history = q_learn_stuff(agent, env)
