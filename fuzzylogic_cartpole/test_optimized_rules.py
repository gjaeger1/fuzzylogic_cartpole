"""
Utility script to decode and test optimized fuzzy rules from EA output
"""

import csv
import sys

import gymnasium as gym
import numpy as np

from .controller import FuzzyCartPoleController
from .defaults import get_standard_domain_specs, get_standard_fuzzy_sets_specs
from .rule_base_generation import (
    generate_rule_base,
    get_standard_domains,
    get_standard_fuzzy_sets,
    save_specification,
)


def decode_genome_from_csv(csv_file, row_index=-1):
    """Load a genome from the CSV output file.

    Args:
        csv_file: Path to the CSV file with optimized genomes
        row_index: Which row to load (-1 for last/best)

    Returns:
        tuple: (genome, fitness) where genome is list of integers
    """
    genomes = []
    fitnesses = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fitness = float(row["fitness"])
            # Parse genome - it's stored as a string representation of a list
            genome_str = row["genome"]
            # Remove brackets and split by comma
            genome_str = genome_str.strip("[]")
            genome = [int(x.strip()) for x in genome_str.split(",")]

            genomes.append(genome)
            fitnesses.append(fitness)

    if not genomes:
        raise ValueError(f"No genomes found in {csv_file}")

    return genomes[row_index], fitnesses[row_index]


def genome_to_rule_specs(genome, domains):
    """Convert genome to rule specifications for saving to YAML.

    Args:
        genome: List of integers (0-4) representing output fuzzy sets
        domains: Tuple of (position, velocity, angle, angular_velocity, action)

    Returns:
        list: Rule specifications
    """
    import itertools

    position, velocity, angle, angular_velocity, action = domains

    # Get all input combinations
    input_domains = [position, velocity, angle, angular_velocity]
    input_fuzzy_sets = []
    for domain in input_domains:
        sets = list(domain._sets.values())
        input_fuzzy_sets.append(sets)

    all_combinations = list(itertools.product(*input_fuzzy_sets))

    # Output fuzzy set names
    output_sets = ["strong_left", "left", "nothing", "right", "strong_right"]

    # Create rule specifications
    rule_specs = []

    for idx, combo in enumerate(all_combinations):
        output_idx = int(genome[idx])
        output_set_name = output_sets[output_idx]

        rule_spec = {
            "output_domain": "action",
            "inputs": [
                f"position.{combo[0]._name}",
                f"velocity.{combo[1]._name}",
                f"angle.{combo[2]._name}",
                f"angular_velocity.{combo[3]._name}",
            ],
            "output": output_set_name,
        }
        rule_specs.append(rule_spec)

    return rule_specs


def test_controller(controller, num_episodes=10, max_steps=500, render=False):
    """Test a fuzzy controller in the CartPole environment.

    Args:
        controller: FuzzyCartPoleController instance
        num_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        dict: Statistics (mean, std, min, max rewards)
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    episode_rewards = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            action = controller.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward:.1f}")

    env.close()

    stats = {
        "mean": np.mean(episode_rewards),
        "std": np.std(episode_rewards),
        "min": np.min(episode_rewards),
        "max": np.max(episode_rewards),
    }

    return stats


def save_optimized_rules_to_yaml(genome, domains, output_file):
    """Save optimized rules to a YAML file.

    Args:
        genome: List of integers representing the rule base
        domains: Tuple of fuzzy domains
        output_file: Path to output YAML file
    """
    # Get standard domain and fuzzy set specifications
    domain_specs = get_standard_domain_specs()
    fuzzy_set_specs = get_standard_fuzzy_sets_specs()

    # Convert genome to rule specifications
    rule_specs = genome_to_rule_specs(genome, domains)

    # Save to YAML
    save_specification(
        domain_specs, fuzzy_set_specs, rule_specs, output_file, default_outputs=None
    )

    print(f"Saved optimized rules to {output_file}")


def main(
    csv_file="optimized_rules.csv",
    yaml_file="optimized_rules.yaml",
    test=True,
    render=False,
):
    """Main function to load, test, and save optimized rules.

    Args:
        csv_file: Input CSV file with genomes from EA
        yaml_file: Output YAML file for the optimized rules
        test: Whether to test the controller
        render: Whether to render the test episodes
    """
    print(f"Loading optimized genome from {csv_file}...")

    # Load best genome
    genome, fitness = decode_genome_from_csv(csv_file)
    print(f"Best fitness from EA: {fitness:.2f}")
    print(f"Genome length: {len(genome)} rules")

    # Setup domains
    print("\nSetting up fuzzy domains and sets...")
    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    domains = (position, velocity, angle, angular_velocity, action)

    # Convert genome to rule specifications
    rule_specs = genome_to_rule_specs(genome, domains)

    # Generate rule base
    rules = generate_rule_base(list(domains), rule_specs, default_outputs=None)

    # Create controller
    controller = FuzzyCartPoleController(domains, rules)

    # Test the controller
    if test:
        print("\nTesting optimized controller...")
        stats = test_controller(
            controller, num_episodes=10, max_steps=500, render=render
        )

        print("\n=== Test Results ===")
        print(f"Mean reward: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"Min reward: {stats['min']:.2f}")
        print(f"Max reward: {stats['max']:.2f}")

    # Save to YAML
    if yaml_file:
        save_optimized_rules_to_yaml(genome, domains, yaml_file)

    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test and save optimized fuzzy rules")
    parser.add_argument("--csv", default="optimized_rules.csv", help="Input CSV file")
    parser.add_argument(
        "--yaml", default="optimized_rules.yaml", help="Output YAML file"
    )
    parser.add_argument("--no-test", action="store_true", help="Skip testing")
    parser.add_argument("--render", action="store_true", help="Render test episodes")

    args = parser.parse_args()

    main(
        csv_file=args.csv,
        yaml_file=args.yaml,
        test=not args.no_test,
        render=args.render,
    )
