"""
Evolutionary Algorithm for Optimizing Fuzzy Rule Base
Uses LEAP library with parallel evaluation via dask
"""

import itertools
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from distributed import Client
from leap_ec import Individual, Representation, ops, probe, test_env_var
from leap_ec.algorithm import generational_ea
from leap_ec.distrib import DistributedIndividual, synchronous
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint
from leap_ec.problem import FunctionProblem
from matplotlib import pyplot as plt

from .controller import FuzzyCartPoleController
from .defaults import get_standard_domain_specs, get_standard_fuzzy_sets_specs
from .rule_base_generation import (
    generate_rule_base,
    get_standard_domains,
    get_standard_fuzzy_sets,
    save_specification,
)

##############################
# Encoding/Decoding Functions
##############################


def get_input_combinations(position, velocity, angle, angular_velocity):
    """Generate all possible combinations of input fuzzy sets.

    Returns:
        list: List of tuples, each containing a combination of input fuzzy sets
    """
    # Get all fuzzy sets for each input domain
    input_domains = [position, velocity, angle, angular_velocity]
    input_fuzzy_sets = []

    for domain in input_domains:
        sets = list(domain._sets.values())
        input_fuzzy_sets.append(sets)

    # Generate all combinations (3 x 3 x 3 x 3 = 81 combinations)
    all_combinations = list(itertools.product(*input_fuzzy_sets))

    return all_combinations


def get_output_fuzzy_sets(action_domain):
    """Get ordered list of output fuzzy sets.

    Returns:
        list: Ordered list of fuzzy set names for the action domain
    """
    # Order: strong_left (0), left (1), nothing (2), right (3), strong_right (4)
    return ["strong_left", "left", "nothing", "right", "strong_right"]


def decode_genome_to_rules(genome, domains):
    """Convert integer genome to rule specifications.

    Args:
        genome: List/array of integers (0-4), one per rule
        domains: Tuple of (position, velocity, angle, angular_velocity, action) domains

    Returns:
        list: Rule specifications suitable for generate_rule_base()
    """
    position, velocity, angle, angular_velocity, action = domains

    # Get all input combinations
    input_combinations = get_input_combinations(
        position, velocity, angle, angular_velocity
    )

    # Get output fuzzy set names
    output_sets = get_output_fuzzy_sets(action)

    # Create rule specifications
    rule_specs = []

    for idx, combo in enumerate(input_combinations):
        # Get the output index from genome
        output_idx = int(genome[idx])
        output_set_name = output_sets[output_idx]

        # Create rule specification
        # combo is (position_set, velocity_set, angle_set, angular_velocity_set)
        rule_spec = {
            "output_domain": "action",
            "inputs": [
                f"position.{combo[0].name}",
                f"velocity.{combo[1].name}",
                f"angle.{combo[2].name}",
                f"angular_velocity.{combo[3].name}",
            ],
            "output": output_set_name,
        }
        rule_specs.append(rule_spec)

    return rule_specs


def genome_to_controller(genome, domains, verbose=False):
    """Convert genome to a FuzzyCartPoleController.

    Args:
        genome: List/array of integers (0-4), one per rule
        domains: Tuple of (position, velocity, angle, angular_velocity, action) domains
        verbose: If True, controller will print debug information

    Returns:
        FuzzyCartPoleController: Controller with rules encoded by genome
    """
    # Decode genome to rule specifications
    rule_specs = decode_genome_to_rules(genome, domains)

    # Generate rule base
    rules = generate_rule_base(list(domains), rule_specs, default_outputs=None)

    # Create controller with verbose=False for efficient EA evaluation
    controller = FuzzyCartPoleController(domains, rules, verbose=verbose)

    return controller


##############################
# Fitness Evaluation
##############################


def evaluate_controller(
    controller, environment, num_episodes=5, max_steps=500, render=False
):
    """Evaluate a fuzzy controller in the CartPole environment.

    Args:
        controller: FuzzyCartPoleController instance
        environment: Gymnasium environment
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        float: Average reward (fitness)
    """
    total_reward = 0.0

    for episode in range(num_episodes):
        observation, info = environment.reset()
        episode_reward = 0

        for step in range(max_steps):
            if render:
                environment.render()

            # Get action from fuzzy controller
            action = controller.get_action(observation)

            # Take step in environment
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_reward += reward - (
                abs(observation[0]) + abs(observation[2])
            )  # subtract positional and angular error

            if terminated or truncated:
                break

        total_reward += episode_reward

    # Return average reward
    return total_reward / num_episodes


def create_fitness_function(num_episodes=5, max_steps=500):
    """Create a fitness function for evaluating genomes.

    Args:
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode

    Returns:
        function: Fitness function that takes a genome and returns fitness
    """

    def fitness_function(genome):
        """Evaluate fitness of a genome."""
        # Recreate domains inside the function to avoid serialization issues
        domains = get_standard_domains()
        position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
            *domains
        )
        full_domains = (position, velocity, angle, angular_velocity, action)

        # Create environment (no rendering for parallel workers)
        env = gym.make("CartPole-v1", render_mode=None)

        # Convert genome to controller
        controller = genome_to_controller(genome, full_domains, verbose=False)

        # Evaluate controller
        fitness = evaluate_controller(
            controller, env, num_episodes, max_steps, render=False
        )

        env.close()

        return fitness

    return fitness_function


##############################
# Probes for Monitoring
##############################


def build_probes(genomes_file=None):
    """Set up probes for writing results to file and terminal."""
    probes = []

    # Print fitness stats to stdout
    probes.append(probe.FitnessStatsCSVProbe(stream=sys.stdout))

    # Save genome of the best individual to a file
    if genomes_file is not None:
        probes.append(
            probe.AttributesCSVProbe(
                stream=genomes_file, best_only=True, do_fitness=True, do_genome=True
            )
        )

    # Plot fitness curve
    plt.figure()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.title("Best-of-Generation Fitness")
    probes.append(
        probe.FitnessPlotProbe(ylim=(0, 500), xlim=(0, 1), modulo=1, ax=plt.gca())
    )

    return probes


##############################
# Initial Population
##############################


def create_initial_population_from_standard(domains, pop_size, num_rules):
    """Create initial population with some individuals based on standard rules.

    Args:
        domains: Tuple of fuzzy domains
        pop_size: Population size
        num_rules: Number of rules (81 for 3x3x3x3 inputs)

    Returns:
        numpy.ndarray: Initial genome with standard rules as numpy array
    """
    from .defaults import get_standard_rules_spec

    # Get standard rule specifications
    standard_specs = get_standard_rules_spec()

    # Create a genome representing standard rules
    position, velocity, angle, angular_velocity, action = domains
    input_combinations = get_input_combinations(
        position, velocity, angle, angular_velocity
    )
    output_sets = get_output_fuzzy_sets(action)

    # Initialize with default (nothing = 2)
    standard_genome = np.full(num_rules, 2, dtype=int)  # nothing is index 2

    # Update with specified rules from standard_specs
    for spec in standard_specs:
        # Find which combination this rule corresponds to
        for idx, combo in enumerate(input_combinations):
            # Check if this combo matches the spec inputs
            if (
                f"position.{combo[0].name}" == spec["inputs"][0]
                and f"velocity.{combo[1].name}" == spec["inputs"][1]
                and f"angle.{combo[2].name}" == spec["inputs"][2]
                and f"angular_velocity.{combo[3].name}" == spec["inputs"][3]
            ):
                # Find output index
                output_name = spec["output"]
                output_idx = output_sets.index(output_name)
                standard_genome[idx] = output_idx
                break

    return standard_genome


##############################
# Main Entry Point
##############################


def optimize_fuzzy_rules(
    pop_size=20,
    generations=50,
    num_episodes=3,
    max_steps=500,
    mutation_rate=0.15,
    use_standard_seed=True,
    output_file="optimized_rules.csv",
    yaml_file="optimized_rules.yaml",
    use_parallel=True,
):
    """Optimize fuzzy rule base using evolutionary algorithm.

    Args:
        pop_size: Population size
        generations: Number of generations
        num_episodes: Episodes per fitness evaluation
        max_steps: Max steps per episode
        mutation_rate: Probability of mutation per gene
        use_standard_seed: Whether to seed population with standard rules
        output_file: File to save best genomes (CSV)
        yaml_file: File to save best rule base (YAML)
        use_parallel: Whether to use parallel evaluation with dask
    """

    # Setup domains and fuzzy sets
    print("Setting up fuzzy domains and sets...")
    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    domains = (position, velocity, angle, angular_velocity, action)

    # Calculate number of rules (3x3x3x3 = 81)
    num_rules = 3 * 3 * 3 * 3
    print(f"Number of rules to optimize: {num_rules}")

    # Create fitness function (doesn't need domains - recreates them internally)
    fitness_func = create_fitness_function(num_episodes, max_steps)

    # Initialize with standard genome if requested
    if use_standard_seed:
        print("Creating initial population with standard rules as seed...")
        standard_genome = create_initial_population_from_standard(
            domains, pop_size, num_rules
        )
    else:
        standard_genome = None

    # Setup the evolutionary algorithm
    print(f"Starting evolutionary optimization...")
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Parallel evaluation: {use_parallel}")

    # Track best individual
    best_individual = {"genome": None, "fitness": float("-inf")}

    def track_best(population):
        """Probe to track the best individual across all generations."""
        best = max(population, key=lambda ind: ind.fitness)
        if best.fitness > best_individual["fitness"]:
            best_individual["genome"] = best.genome.copy()
            best_individual["fitness"] = best.fitness
        return population

    def run_evolution(client=None):
        """Run the evolutionary algorithm."""
        with open(output_file, "w") as genomes_file:
            # Create initializer
            if standard_genome is not None:
                # Custom initializer that includes standard genome
                def custom_initializer():
                    # First individual is the standard genome (convert to numpy array)
                    yield np.array(standard_genome, dtype=int)
                    # Rest are random
                    for _ in range(pop_size - 1):
                        yield np.random.randint(0, 5, size=num_rules)

                init_func = custom_initializer()
            else:
                init_func = create_int_vector(bounds=[(0, 4)] * num_rules)

            # Build the pipeline
            pipeline = [
                ops.tournament_selection,
                ops.clone,
                mutate_randint(
                    bounds=np.array([[0, 4]] * num_rules),
                    expected_num_mutations=int(num_rules * mutation_rate),
                ),
                ops.UniformCrossover(p_swap=0.1),
            ]

            # Add evaluation operator
            if use_parallel and client is not None:
                pipeline.append(synchronous.eval_pool(client=client, size=pop_size))
            else:
                pipeline.append(ops.evaluate)

            # Add probes
            pipeline.extend(build_probes(genomes_file))
            pipeline.append(track_best)

            # Run EA
            generational_ea(
                max_generations=generations,
                pop_size=pop_size,
                problem=FunctionProblem(fitness_func, maximize=True),
                representation=Representation(
                    initialize=init_func
                    if standard_genome is None
                    else lambda: next(init_func),
                    individual_cls=DistributedIndividual
                    if use_parallel
                    else Individual,
                ),
                pipeline=pipeline,
            )

    # Run with or without dask
    if use_parallel:
        print("Starting dask client for parallel fitness evaluations...")
        with Client() as client:
            print(f"Dask dashboard available at: {client.dashboard_link}")
            print(f"Number of workers: {len(client.scheduler_info()['workers'])}")
            run_evolution(client)
    else:
        run_evolution()

    print(f"\nOptimization complete! Best genomes saved to {output_file}")

    # Save best rule base to YAML
    if best_individual["genome"] is not None:
        print(f"\nSaving best rule base to {yaml_file}...")
        print(f"Best fitness: {best_individual['fitness']:.2f}")

        # Convert genome to rule specifications
        best_genome = best_individual["genome"]

        # Get domain and fuzzy set specifications
        domain_specs = get_standard_domain_specs()
        fuzzy_set_specs = get_standard_fuzzy_sets_specs()

        # Convert genome to rule specifications
        rule_specs = decode_genome_to_rules(best_genome, domains)

        # Save to YAML
        save_specification(
            domain_specs,
            fuzzy_set_specs,
            rule_specs,
            yaml_file,
            default_outputs=None,
        )

        print(f"Best rule base saved to {yaml_file}")
    else:
        print("\nWarning: No best individual found to save.")

    # Show plots if not in test mode
    if os.environ.get(test_env_var, False) != "True":
        plt.show()

    plt.close("all")


##############################
# Command-line interface
##############################

if __name__ == "__main__":
    # When running the test harness, just run for two generations
    if os.environ.get(test_env_var, False) == "True":
        generations = 2
        pop_size = 5
    else:
        generations = 50
        pop_size = 20

    optimize_fuzzy_rules(
        pop_size=pop_size,
        generations=generations,
        num_episodes=3,
        max_steps=500,
        mutation_rate=0.15,
        use_standard_seed=True,
        output_file="optimized_rules.csv",
        yaml_file="optimized_rules.yaml",
        use_parallel=True,  # Now safe for parallel execution
    )
