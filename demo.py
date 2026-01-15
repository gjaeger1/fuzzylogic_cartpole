"""
Demo script to run the Fuzzy Logic CartPole controller.

This script creates a CartPole-v1 environment and runs it using the fuzzy logic controller.
"""

import os

import click
import gymnasium as gym

from fuzzylogic_cartpole import (
    FuzzyCartPoleController,
)
from fuzzylogic_cartpole.rule_base_generation import (
    generate_domains,
    generate_fuzzy_sets,
    generate_rule_base,
    get_standard_rules,
    get_standard_specifications,
    load_specification,
    save_specification,
)


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
    observation, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if render:
            env.render()

        # Get action from fuzzy controller
        action = controller.get_action(observation)

        # Take action in environment
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


@click.command()
@click.option(
    "--config",
    type=click.Path(),
    default="fuzzy_config.yaml",
    help="Path to the YAML configuration file for the fuzzy controller.",
)
def main(config):
    """Main function to run the demo."""
    # Check if config file exists, if not create it with standard configuration
    if not os.path.exists(config):
        print(
            f"Configuration file '{config}' not found. Creating with standard configuration..."
        )
        domain_specs, fuzzy_set_specs, rule_specs = get_standard_specifications()
        save_specification(domain_specs, fuzzy_set_specs, rule_specs, config)
        print(f"Created configuration file: {config}")

    # Load configuration
    print(f"Loading configuration from: {config}")
    domain_specs, fuzzy_set_specs, rule_specs = load_specification(config)

    # Generate domains, fuzzy sets, and rules
    domains = generate_fuzzy_sets(generate_domains(domain_specs), fuzzy_set_specs)
    position, velocity, angle, angular_velocity, action = domains
    rules = generate_rule_base(
        domains,
        rule_specs,
        {"action": "nothing"},
    )

    # Create environment with visualization enabled
    env = gym.make("CartPole-v1", render_mode="human")

    # Create fuzzy controller with loaded configuration
    controller = FuzzyCartPoleController(domains, rules)

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
