import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from EnvAgt import ParkinEnv, Agent_007
import pandas as pd
from tqdm import tqdm

class RLEvaluator:
    """
    Evaluator class for RL models for the automatic parking problem
    """
    def __init__(self):
        self.results = []

    def evaluate_policy(self, agent, env, n_eval_episodes=100, max_steps=200):
        """
        Evaluates a trained policy by running multiple episodes.
        Returns average reward, success rate, and state coverage.
        """
        total_rewards = []
        success_rate = 0
        steps_to_goal = []
        visited_states = set()
        for _ in range(n_eval_episodes):
            env.reset()
            state = env.s0
            episode_reward = 0
            steps = 0
            done = False
            # one episode
            while not done and steps < max_steps:
                action_idx = agent.choose_action(state)
                action = agent.actions[action_idx]
                next_state, reward = env.interact(action)
                episode_reward += reward
                state = next_state
                steps += 1
                visited_states.add(state)
                done = reward == env.pos_reward
            if done:
                success_rate += 1
                steps_to_goal.append(steps)
            total_rewards.append(episode_reward)
        return {
            'mean_reward': np.mean(total_rewards),
            'success_rate': success_rate / n_eval_episodes,
            'mean_steps': np.mean(steps_to_goal) if steps_to_goal else max_steps,
            'std_reward': np.std(total_rewards),
            'n_states_visited_eval': len(visited_states)
        }

    def train_and_evaluate(self, discretize_val, learning_rate, discount_factor, n_episodes,
                           max_episode_length, epsilon_greedy=0.9):
        """
        Runs a full training with the specified parameters and evaluates the resulting policy.
        
        """
        env = ParkinEnv(discretize_val=discretize_val)
        agent = Agent_007(env, learning_rate=learning_rate, discount_factor=discount_factor,
                         epsilon_greedy=epsilon_greedy)
        eval_interval = n_episodes // 10
        for episode in tqdm(range(n_episodes), desc=f"Training (disc={discretize_val}, lr={learning_rate})"):
            env.reset()
            state = env.s0
            steps = 0
            done = False
            while not done and steps < max_episode_length:
                action_idx = agent.choose_action(state)
                action = agent.actions[action_idx]
                next_state, reward = env.interact(action)
                done = reward == env.pos_reward
                # update q-vals
                agent._learn((state, action_idx, reward, next_state, done))
                state = next_state
                steps += 1
        # evaluates the final policy
        final_metrics = self.evaluate_policy(agent, env, n_eval_episodes=200, max_steps=max_episode_length)
        q_stats = self.analyze_q_table(agent.q_table)
        result = {
            'discretization': discretize_val,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'episodes': n_episodes,
            'max_length': max_episode_length,
            **final_metrics,
            **q_stats
        }
        self.results.append(result)
        return agent, result

    def analyze_q_table(self, q_table):
        """
        Q-table characteristics for further analysis
        Returns stats properties and state coverage
        """
        q_values = np.array([list(values) for values in q_table.values()])
        if len(q_values) == 0:
            return {'empty': True}
        return {
            'mean_q': np.mean(q_values),
            'std_q': np.std(q_values),
            'max_q': np.max(q_values),
            'min_q': np.min(q_values),
            'n_states_visited': len(q_table)
        }

    def print_summary(self):
        """
        Displays a summary table of all experiments for later comparison.
        """
        df = pd.DataFrame(self.results)
        print("\nSummary of all experiments:")
        print(df.to_string(index=False))

    def plot_trajectory(self, agent, env, max_steps=500, delta=0.1, epsilon=None):
        """
        Visualizes the car's trajectory using the trained policy.
        """
        env.reset()
        state = env.s0
        positions = [state[0]]
        velocities = [state[1]]
        times = [0]
        t = 0
        # temp override agent epsilon if not None
        if epsilon is not None:
            old_eps = agent.eps
            agent.eps = epsilon
        for step in range(max_steps):
            action_idx = agent.choose_action(state)
            action = agent.actions[action_idx]
            next_state, reward = env.interact(action)
            t += delta
            positions.append(next_state[0])
            velocities.append(next_state[1])
            times.append(t)
            state = next_state
            if reward == env.pos_reward:
                break
        if epsilon is not None:
            agent.eps = old_eps  # Restore original epsilon
        plt.figure(figsize=(8, 5))
        plt.plot(times, positions, 'r-', label='Position')
        plt.plot(times, velocities, 'b-', label='Velocity')
        plt.xlabel('t')
        plt.ylabel('Value')
        plt.title('Trajectory of the car with the evaluated policy')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    eval = RLEvaluator()
    config = {
        'discretize_val': 51,
        'learning_rate': 0.01,
        'discount_factor': 0.9,
        'n_episodes': 10000,
        'max_episode_length': 500
    }
    print("Running one configuration:")
    print(f"Parameters: {config}")
    agent, result = eval.train_and_evaluate(**config)
    print("\nResults:")
    for k, v in result.items():
        print(f"{k}: {v}")
    eval.plot_trajectory(agent, ParkinEnv(discretize_val=config['discretize_val']), 
                              max_steps=config['max_episode_length']) 