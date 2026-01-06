"""
Demo script to run the Fuzzy Logic CartPole controller.

This script creates a CartPole-v1 environment and runs it using the fuzzy logic controller.
"""

import gymnasium as gym

from fuzzylogic_cartpole import FuzzyCartPoleController


def run_episode(env, controller, render=False):
    """
    Run a single episode of CartPole with the fuzzy controller.

    Args:
        env: The Gymnasium environment
        controller: The FuzzyCartPoleController instance
        render: Whether to render the environment

    Returns:
        int: Total reward (number of timesteps the pole stayed upright)
    """
    observation, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        if render:
            env.render()

        # Get action from fuzzy controller
        action = controller.get_action(observation)

        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def main():
    """Main function to run the demo."""
    # Create environment with visualization enabled
    env = gym.make("CartPole-v1", render_mode="human")

    # Create fuzzy controller
    controller = FuzzyCartPoleController()

    # Run multiple episodes
    num_episodes = 1
    rewards = []

    print(f"Running {num_episodes} episodes with Fuzzy Logic Controller...")
    print("-" * 50)

    for episode in range(num_episodes):
        reward = run_episode(env, controller, render=True)
        rewards.append(reward)
        print(f"Episode {episode + 1}: {reward} timesteps")

    print("-" * 50)
    print(f"Average reward: {sum(rewards) / len(rewards):.2f} timesteps")
    print(f"Best reward: {max(rewards)} timesteps")
    print(f"Worst reward: {min(rewards)} timesteps")

    env.close()


if __name__ == "__main__":
    main()
