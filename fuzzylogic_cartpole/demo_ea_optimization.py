"""
Quick-Start Demo: Evolutionary Optimization of Fuzzy Rules
============================================================

This demo shows the minimal necessary code to optimize fuzzy rules using
evolutionary algorithms with the LEAP library.

Key concepts:
1. Encoding: Each rule base is a vector of 81 integers (0-4)
2. Decoding: Convert integers to fuzzy rule specifications
3. Fitness: Evaluate controller performance in CartPole
4. Evolution: Use LEAP to find better rule bases
"""

import sys

# Import gymnasium for CartPole environment
import gymnasium as gym
import numpy as np
from leap_ec import Individual, Representation, ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint

from .controller import FuzzyCartPoleController

# Import fuzzy logic components
from .rule_base_generation import (
    generate_rule_base,
    get_standard_domains,
    get_standard_fuzzy_sets,
)


def demo():
    """
    Minimal demonstration of EA-based fuzzy rule optimization.

    This function shows the essential steps:
    1. Setup fuzzy domains and sets
    2. Define encoding (genome -> rules)
    3. Create fitness function
    4. Run evolutionary algorithm
    5. Test the best solution
    """

    print("=" * 70)
    print("FUZZY RULE OPTIMIZATION DEMO")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Setup Fuzzy Logic System
    # ========================================================================
    print("\nStep 1: Setting up fuzzy logic domains...")

    # Create standard domains (position, velocity, angle, angular_velocity, action)
    domains = get_standard_domains()

    # Add fuzzy sets to each domain (negative, zero, positive for inputs)
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    domains = (position, velocity, angle, angular_velocity, action)

    # Calculate number of rules: 3 x 3 x 3 x 3 = 81
    # (each input has 3 fuzzy sets: negative, zero, positive)
    num_rules = 3 * 3 * 3 * 3
    print(f"   Number of rules: {num_rules}")
    print(f"   Input domains: 4 (position, velocity, angle, angular_velocity)")
    print(f"   Output domain: 1 (action with 5 fuzzy sets)")

    # ========================================================================
    # STEP 2: Define Genome-to-Controller Decoder
    # ========================================================================
    print("\nStep 2: Defining genome encoding...")
    print(f"   Genome: Vector of {num_rules} integers")
    print(f"   Each integer (0-4) represents an output fuzzy set:")
    print(f"      0 = strong_left")
    print(f"      1 = left")
    print(f"      2 = nothing")
    print(f"      3 = right")
    print(f"      4 = strong_right")

    def decode_genome(genome):
        """
        Convert genome (list of integers) to FuzzyCartPoleController.

        This is the core decoder that maps genotype to phenotype.
        """
        import itertools

        # Get all combinations of input fuzzy sets
        input_domains = [position, velocity, angle, angular_velocity]
        input_sets = [list(d._sets.values()) for d in input_domains]
        combinations = list(itertools.product(*input_sets))

        # Output fuzzy set names (indexed 0-4)
        output_names = ["strong_left", "left", "nothing", "right", "strong_right"]

        # Build rule specifications
        rule_specs = []
        for idx, combo in enumerate(combinations):
            output_idx = int(genome[idx])
            rule_spec = {
                "output_domain": "action",
                "inputs": [
                    f"position.{combo[0]._name}",
                    f"velocity.{combo[1]._name}",
                    f"angle.{combo[2]._name}",
                    f"angular_velocity.{combo[3]._name}",
                ],
                "output": output_names[output_idx],
            }
            rule_specs.append(rule_spec)

        # Generate rules and create controller
        rules = generate_rule_base(list(domains), rule_specs)
        return FuzzyCartPoleController(domains, rules, verbose=False)

    # ========================================================================
    # STEP 3: Define Fitness Function
    # ========================================================================
    print("\nStep 3: Creating fitness function...")

    def evaluate_fitness(genome):
        """
        Fitness function: Evaluate how well a genome performs in CartPole.

        Returns average reward over multiple episodes.
        """
        # Create controller from genome
        controller = decode_genome(genome)

        # Create environment
        env = gym.make("CartPole-v1", render_mode=None)

        # Run multiple episodes and average the rewards
        num_episodes = 3
        max_steps = 500
        total_reward = 0.0

        for _ in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                # Get action from fuzzy controller
                action = controller.get_action(observation)

                # Take step in environment
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_reward += episode_reward

        env.close()

        # Return average reward as fitness
        return total_reward / num_episodes

    print(f"   Episodes per evaluation: 3")
    print(f"   Max steps per episode: 500")

    # ========================================================================
    # STEP 4: Run Evolutionary Algorithm
    # ========================================================================
    print("\nStep 4: Running evolutionary algorithm...")

    # EA parameters
    pop_size = 10
    generations = 5
    mutation_rate = 0.15

    print(f"   Population size: {pop_size}")
    print(f"   Generations: {generations}")
    print(f"   Mutation rate: {mutation_rate}")

    # Track best individual across generations
    best_ever = {"genome": None, "fitness": float("-inf")}

    def track_best(population):
        """Probe to track the best individual."""
        best = max(population, key=lambda ind: ind.fitness)
        if best.fitness > best_ever["fitness"]:
            best_ever["genome"] = best.genome[:]
            best_ever["fitness"] = best.fitness
        return population

    # Run the EA
    print("\n" + "-" * 70)
    final_pop = generational_ea(
        max_generations=generations,
        pop_size=pop_size,
        # Fitness function
        problem=evaluate_fitness,
        # Representation: integer vectors with values 0-4
        representation=Representation(
            initialize=create_int_vector(bounds=[(0, 4)] * num_rules)
        ),
        # EA pipeline: selection -> clone -> mutate -> crossover -> evaluate
        pipeline=[
            ops.tournament_selection,  # Select parents via tournament
            ops.clone,  # Clone for variation
            mutate_randint(  # Mutate some genes
                bounds=(0, 4), expected_num_mutations=int(num_rules * mutation_rate)
            ),
            ops.UniformCrossover(p_swap=0.1),  # Crossover with 10% swap probability
            ops.evaluate,  # Evaluate fitness
            probe.FitnessStatsCSVProbe(stream=sys.stdout),  # Print stats
            track_best,  # Track best solution
        ],
    )
    print("-" * 70)

    # ========================================================================
    # STEP 5: Test Best Solution
    # ========================================================================
    print("\nStep 5: Testing best solution...")

    if best_ever["genome"] is not None:
        print(f"\n   Best fitness found: {best_ever['fitness']:.2f}")

        # Create controller from best genome
        best_controller = decode_genome(best_ever["genome"])

        # Test with more episodes
        print("\n   Running 10 test episodes...")
        env = gym.make("CartPole-v1", render_mode=None)
        test_rewards = []

        for ep in range(10):
            observation, _ = env.reset()
            episode_reward = 0

            for _ in range(500):
                action = best_controller.get_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            test_rewards.append(episode_reward)
            print(f"      Episode {ep + 1}: {episode_reward:.1f}")

        env.close()

        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)

        print(f"\n   Test Results:")
        print(f"      Mean: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"      Min:  {np.min(test_rewards):.2f}")
        print(f"      Max:  {np.max(test_rewards):.2f}")

        # Show example genome
        print(f"\n   Best genome (first 10 rules): {best_ever['genome'][:10]}")
        print(f"   (Complete genome has {num_rules} integers)")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Each genome is a vector of 81 integers (one per rule)")
    print("  • Integers encode which output fuzzy set to use (0-4)")
    print("  • EA evolves better rule bases by mutating/crossing genomes")
    print("  • Fitness is measured by CartPole performance")
    print(
        "\nFor full optimization, run: python -m fuzzylogic_cartpole.optimize_rules_ea"
    )


if __name__ == "__main__":
    demo()
