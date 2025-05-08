from rl_eval import RLEvaluator
import numpy as np
import itertools

def main():
    eval = RLEvaluator()
    
    # param grid for comprehensive testing asked in task 1.5
    discretization_values = [21, 51, 101]
    learning_rates = [0.01, 0.1]
    discount_factors = [0.9, 0.99]
    episode_configs = [
        (5000, 100),
        (10000, 500),
        (20000, 1000)
    ]
    for disc_val, lr, gamma, (n_episodes, max_length) in itertools.product(
            discretization_values, learning_rates, discount_factors, episode_configs):
        print(f"\nRunning configuration: d={disc_val}, lr={lr}, gamma={gamma}, episodes={n_episodes}, max_len={max_length}")
        agent, result = eval.train_and_evaluate(
            discretize_val=disc_val,
            learning_rate=lr,
            discount_factor=gamma,
            n_episodes=n_episodes,
            max_episode_length=max_length
        )
        print(f"Result: Success Rate={result['success_rate']:.2%}, Mean Steps={result['mean_steps']:.2f}, Mean Reward={result['mean_reward']:.2f}, Q-table States={result['n_states_visited']}")
    eval.print_summary()

if __name__ == "__main__":
    main() 