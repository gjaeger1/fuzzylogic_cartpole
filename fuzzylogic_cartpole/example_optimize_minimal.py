"""
Minimal example for optimizing fuzzy rules with evolutionary algorithms.
This is a simplified version for quick testing with fewer generations.
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from distributed import Client
from leap_ec import Individual, Representation, ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.distrib import DistributedIndividual, synchronous
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint
from matplotlib import pyplot as plt

# Import from the fuzzylogic_cartpole package
try:
    from fuzzylogic_cartpole.optimize_rules_ea import (
        create_fitness_function,
        create_initial_population_from_standard,
        genome_to_controller,
    )
    from fuzzylogic_cartpole.rule_base_generation import (
        get_standard_domains,
        get_standard_fuzzy_sets,
    )
except ImportError:
    # Alternative import for when running as script
    from optimize_rules_ea import (
        create_fitness_function,
        create_initial_population_from_standard,
        genome_to_controller,
    )
    from rule_base_generation import get_standard_domains, get_standard_fuzzy_sets


def minimal_optimize():
    """Minimal EA optimization example with reduced parameters for quick testing."""

    print("=" * 60)
    print("Minimal Fuzzy Rule Optimization Example")
    print("=" * 60)

    # Minimal parameters for quick testing
    POP_SIZE = 10
    GENERATIONS = 10
    NUM_EPISODES = 2
    MAX_STEPS = 300
    MUTATION_RATE = 0.2
    USE_PARALLEL = False  # Set to True to enable parallel evaluation

    # Setup domains and fuzzy sets
    print("\n1. Setting up fuzzy logic domains...")
    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    domains = (position, velocity, angle, angular_velocity, action)

    # Number of rules (3x3x3x3 = 81)
    num_rules = 81
    print(f"   Total rules to optimize: {num_rules}")

    # Create fitness function
    print("\n2. Creating fitness function...")
    fitness_func = create_fitness_function(domains, NUM_EPISODES, MAX_STEPS)

    # Get standard genome as starting point
    print("\n3. Creating initial population with standard rules...")
    standard_genome = create_initial_population_from_standard(
        domains, POP_SIZE, num_rules
    )

    # Test standard genome
    print("\n4. Evaluating standard rule base...")
    standard_fitness = fitness_func(standard_genome)
    print(f"   Standard rules fitness: {standard_fitness:.2f}")

    # Setup EA
    print(f"\n5. Starting evolutionary optimization...")
    print(f"   Population size: {POP_SIZE}")
    print(f"   Generations: {GENERATIONS}")
    print(f"   Mutation rate: {MUTATION_RATE}")
    print(f"   Parallel evaluation: {USE_PARALLEL}")

    # Custom initializer that includes standard genome
    def custom_initializer():
        """Generate initial population with standard genome as seed."""
        yield standard_genome
        for _ in range(POP_SIZE - 1):
            yield np.random.randint(0, 5, size=num_rules).tolist()

    init_func = custom_initializer()

    # Build pipeline
    pipeline = [
        ops.tournament_selection,
        ops.clone,
        mutate_randint(
            bounds=(0, 4), expected_num_mutations=int(num_rules * MUTATION_RATE)
        ),
        ops.UniformCrossover(p_swap=0.1),
    ]

    # Add evaluation
    if USE_PARALLEL:
        print("\n   Starting dask client...")
        client = Client()
        print(f"   Dask dashboard: {client.dashboard_link}")
        pipeline.append(synchronous.eval_pool(client=client, size=POP_SIZE))
    else:
        pipeline.append(ops.evaluate)

    # Add simple probes
    pipeline.append(probe.FitnessStatsCSVProbe(stream=sys.stdout))

    # Track best individual
    best_individual = []

    def capture_best(population):
        """Probe to capture the best individual."""
        best = max(population, key=lambda ind: ind.fitness)
        best_individual.append((best.genome[:], best.fitness))
        return population

    pipeline.append(capture_best)

    # Run EA
    try:
        final_pop = generational_ea(
            max_generations=GENERATIONS,
            pop_size=POP_SIZE,
            problem=fitness_func,
            representation=Representation(
                initialize=lambda: next(init_func),
                individual_cls=DistributedIndividual if USE_PARALLEL else Individual,
            ),
            pipeline=pipeline,
        )

        # Get best result
        if best_individual:
            best_genome, best_fitness = best_individual[-1]
            print(f"\n" + "=" * 60)
            print(f"OPTIMIZATION COMPLETE!")
            print(f"=" * 60)
            print(f"Best fitness: {best_fitness:.2f}")
            print(f"Improvement: {best_fitness - standard_fitness:.2f}")
            print(
                f"Relative improvement: {((best_fitness - standard_fitness) / standard_fitness * 100):.1f}%"
            )

            # Show how many rules changed
            num_changes = sum(
                1 for i in range(num_rules) if best_genome[i] != standard_genome[i]
            )
            print(f"Rules modified: {num_changes}/{num_rules}")

            # Save best genome
            output_file = "minimal_optimized_rules.txt"
            with open(output_file, "w") as f:
                f.write(f"# Best fitness: {best_fitness:.2f}\n")
                f.write(f"# Genome (81 integers, 0-4):\n")
                f.write(str(best_genome))
            print(f"\nBest genome saved to: {output_file}")

            # Test the best controller
            print("\n6. Testing optimized controller (5 episodes)...")
            env = gym.make("CartPole-v1", render_mode=None)
            controller = genome_to_controller(best_genome, domains)

            test_rewards = []
            for ep in range(5):
                obs, _ = env.reset()
                total_reward = 0
                for step in range(500):
                    action = controller.get_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(total_reward)
                print(f"   Episode {ep + 1}: {total_reward:.1f}")

            env.close()
            print(
                f"\n   Test average: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}"
            )

    finally:
        if USE_PARALLEL:
            client.close()

    print("\nDone!")


if __name__ == "__main__":
    minimal_optimize()
