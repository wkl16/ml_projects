import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from environment_22 import Environment
from agent_234 import Agent
from map_abstraction_21 import Map_Abstraction
np.random.seed(1)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def run_qlearning(agent, env, num_episodes=50):
    history = []
    for episode in range(num_episodes):
        state = env.reset() # initial state
        final_reward, n_moves = 0.0, 0
        done = False

        while not done:
            action_idx = agent.choose_action(state)
            action = agent.actions[action_idx]
            next_state, reward = env.interaction(state.copy(), action)
            done = reward == env.goal_reward  # goal reached
            agent.learn(Transition(state, action_idx, reward, next_state, done))
            state = next_state.copy()
            n_moves += 1
            final_reward = reward  # store final reward for logging

        history.append((n_moves, final_reward))
        print(f'Episode {episode}: Reward {final_reward:.2f}, #Moves {n_moves}')
    return history

def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('# Moves', size=20)

    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Final rewards', size=20)

    plt.tight_layout()
    plt.savefig('q-learning-history.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    ma = Map_Abstraction()
   #np_img_array = ma.abstract("./maps/map1.bmp", 20, 20)
    np_img_array = ma.abstract("./maps/map2.bmp", 40, 40)
    env = Environment(np_img_array, 0, 0, 38, 38)
    agent = Agent(env)
    history = run_qlearning(agent, env)
    plot_learning_history(history)
