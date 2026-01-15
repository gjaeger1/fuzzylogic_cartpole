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
from .rule_base_generation import (
    generate_domains,
    generate_fuzzy_sets,
    generate_rule_base,
    save_specification,
)

##############################
# Encoding/Decoding Functions
##############################


def get_input_combinations(domains, input_domain_names):
    """Generate all possible combinations of input fuzzy sets.

    Args:
        domains: Tuple of all Domain objects
        input_domain_names: List of input domain names (e.g., ['position', 'velocity', 'angle', 'angular_velocity'])

    Returns:
        list: List of tuples, each containing a combination of input fuzzy sets
    """
    # Create domain dict for lookup
    domain_dict = {d._name: d for d in domains}

    # Get all fuzzy sets for each input domain
    input_fuzzy_sets = []
    for domain_name in input_domain_names:
        domain = domain_dict[domain_name]
        sets = list(domain._sets.values())
        input_fuzzy_sets.append(sets)

    # Generate all combinations
    all_combinations = list(itertools.product(*input_fuzzy_sets))

    return all_combinations


def get_output_fuzzy_sets(output_domain):
    """Get ordered list of output fuzzy sets from a domain.

    Args:
        output_domain: The output Domain object

    Returns:
        list: Ordered list of fuzzy set names for the output domain
    """
    # Return all fuzzy set names in the order they were defined
    return list(output_domain._sets.keys())


def decode_genome_to_rules(genome, domains, input_domain_names, output_domain_name):
    """Convert integer genome to rule specifications.

    Args:
        genome: List/array of integers, one per rule
        domains: Tuple of all Domain objects
        input_domain_names: List of input domain names in order
        output_domain_name: Name of the output domain

    Returns:
        list: Rule specifications suitable for generate_rule_base()
    """
    # Create domain dict for lookup
    domain_dict = {d._name: d for d in domains}

    # Get output domain
    output_domain = domain_dict[output_domain_name]

    # Get all input combinations
    input_combinations = get_input_combinations(domains, input_domain_names)

    # Get output fuzzy set names
    output_sets = get_output_fuzzy_sets(output_domain)

    # Create rule specifications
    rule_specs = []

    for idx, combo in enumerate(input_combinations):
        # Get the output index from genome
        output_idx = int(genome[idx])
        output_set_name = output_sets[output_idx]

        # Create rule specification
        rule_spec = {
            "output_domain": output_domain_name,
            "inputs": [
                f"{input_domain_names[i]}.{combo[i].name}"
                for i in range(len(input_domain_names))
            ],
            "output": output_set_name,
        }
        rule_specs.append(rule_spec)

    return rule_specs


def genome_to_controller(
    genome, domains, input_domain_names, output_domain_name, verbose=False
):
    """Convert genome to a FuzzyCartPoleController.

    Args:
        genome: List/array of integers, one per rule
        domains: Tuple of all Domain objects
        input_domain_names: List of input domain names in order
        output_domain_name: Name of the output domain
        verbose: If True, controller will print debug information

    Returns:
        FuzzyCartPoleController: Controller with rules encoded by genome
    """
    # Decode genome to rule specifications
    rule_specs = decode_genome_to_rules(
        genome, domains, input_domain_names, output_domain_name
    )

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


def create_fitness_function(
    domain_specs,
    fuzzy_set_specs,
    input_domain_names,
    output_domain_name,
    num_episodes=5,
    max_steps=500,
):
    """Create a fitness function for evaluating genomes.

    Args:
        domain_specs: List of domain specifications (for generate_domains)
        fuzzy_set_specs: List of fuzzy set specifications (for generate_fuzzy_sets)
        input_domain_names: List of input domain names in order
        output_domain_name: Name of the output domain
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode

    Returns:
        function: Fitness function that takes a genome and returns fitness
    """

    def fitness_function(genome):
        """Evaluate fitness of a genome."""
        # Recreate domains inside the function to avoid serialization issues
        domains = generate_domains(domain_specs)
        domains = generate_fuzzy_sets(domains, fuzzy_set_specs)

        # Create environment (no rendering for parallel workers)
        env = gym.make("CartPole-v1", render_mode=None)

        # Convert genome to controller
        controller = genome_to_controller(
            genome, domains, input_domain_names, output_domain_name, verbose=False
        )

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


def create_initial_genome_from_specs(
    domains,
    input_domain_names,
    output_domain_name,
    initial_rule_specs,
    default_output_name,
    num_rules,
):
    """Create initial genome based on provided rule specifications.

    Args:
        domains: Tuple of all Domain objects
        input_domain_names: List of input domain names in order
        output_domain_name: Name of the output domain
        initial_rule_specs: List of rule specifications to use as starting point
        default_output_name: Name of default output fuzzy set for unspecified rules
        num_rules: Total number of rules

    Returns:
        numpy.ndarray: Initial genome as numpy array
    """
    # Create domain dict for lookup
    domain_dict = {d._name: d for d in domains}
    output_domain = domain_dict[output_domain_name]

    # Get all input combinations
    input_combinations = get_input_combinations(domains, input_domain_names)
    output_sets = get_output_fuzzy_sets(output_domain)

    # Initialize with default output
    default_idx = output_sets.index(default_output_name)
    genome = np.full(num_rules, default_idx, dtype=int)

    # Update with specified rules
    for spec in initial_rule_specs:
        # Find which combination this rule corresponds to
        for idx, combo in enumerate(input_combinations):
            # Check if this combo matches the spec inputs
            inputs_match = all(
                f"{input_domain_names[i]}.{combo[i].name}" == spec["inputs"][i]
                for i in range(len(input_domain_names))
            )

            if inputs_match:
                # Find output index
                output_name = spec["output"]
                output_idx = output_sets.index(output_name)
                genome[idx] = output_idx
                break

    return genome


##############################
# Main Entry Point
##############################


def optimize_fuzzy_rules(
    domain_specs,
    fuzzy_set_specs,
    input_domain_names,
    output_domain_name,
    pop_size=20,
    generations=50,
    num_episodes=3,
    max_steps=500,
    mutation_rate=0.15,
    use_initial_seed=False,
    initial_rule_specs=None,
    default_output_name=None,
    output_file="optimized_rules.csv",
    yaml_file="optimized_rules.yaml",
    use_parallel=True,
):
    """Optimize fuzzy rule base using evolutionary algorithm.

    Args:
        domain_specs: List of domain specifications (for generate_domains)
        fuzzy_set_specs: List of fuzzy set specifications (for generate_fuzzy_sets)
        input_domain_names: List of input domain names in order (e.g., ['position', 'velocity', 'angle', 'angular_velocity'])
        output_domain_name: Name of the output domain (e.g., 'action')
        pop_size: Population size
        generations: Number of generations
        num_episodes: Episodes per fitness evaluation
        max_steps: Max steps per episode
        mutation_rate: Probability of mutation per gene
        use_initial_seed: If True, use initial_rule_specs to seed first individual; if False, use random initialization
        initial_rule_specs: Optional list of rule specifications to seed first individual (used if use_initial_seed=True)
        default_output_name: Default output fuzzy set name for unseeded rules (required if use_initial_seed=True)
        output_file: File to save best genomes (CSV)
        yaml_file: File to save best rule base (YAML)
        use_parallel: Whether to use parallel evaluation with dask
    """

    # Setup domains and fuzzy sets
    print("Setting up fuzzy domains and sets...")
    domains = generate_domains(domain_specs)
    domains = generate_fuzzy_sets(domains, fuzzy_set_specs)

    # Create domain dict for calculations
    domain_dict = {d._name: d for d in domains}

    # Calculate number of rules based on input domain sizes
    num_rules = 1
    for domain_name in input_domain_names:
        domain = domain_dict[domain_name]
        num_rules *= len(domain._sets)
    print(f"Number of rules to optimize: {num_rules}")

    # Get number of output fuzzy sets
    output_domain = domain_dict[output_domain_name]
    num_outputs = len(output_domain._sets)
    print(f"Number of output fuzzy sets: {num_outputs}")

    # Create fitness function
    fitness_func = create_fitness_function(
        domain_specs,
        fuzzy_set_specs,
        input_domain_names,
        output_domain_name,
        num_episodes,
        max_steps,
    )

    # Initialize with provided rule specs if requested
    if use_initial_seed:
        if initial_rule_specs is None:
            raise ValueError(
                "initial_rule_specs must be provided when use_initial_seed=True"
            )
        if default_output_name is None:
            raise ValueError(
                "default_output_name must be provided when use_initial_seed=True"
            )
        print(
            "Creating initial population with provided rule specifications as seed..."
        )
        initial_genome = create_initial_genome_from_specs(
            domains,
            input_domain_names,
            output_domain_name,
            initial_rule_specs,
            default_output_name,
            num_rules,
        )
    else:
        print("Creating initial population with random genomes...")
        initial_genome = None

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
            if initial_genome is not None:
                # Custom initializer that includes initial genome
                def custom_initializer():
                    # First individual is the initial genome
                    yield np.array(initial_genome, dtype=int)
                    # Rest are random
                    for _ in range(pop_size - 1):
                        yield np.random.randint(0, num_outputs, size=num_rules)

                init_func = custom_initializer()
            else:
                init_func = create_int_vector(bounds=[(0, num_outputs - 1)] * num_rules)

            # Build the pipeline
            pipeline = [
                ops.tournament_selection,
                ops.clone,
                mutate_randint(
                    bounds=np.array([[0, num_outputs - 1]] * num_rules),
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
                    if initial_genome is None
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

        # Convert genome to rule specifications
        rule_specs = decode_genome_to_rules(
            best_genome, domains, input_domain_names, output_domain_name
        )

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
    # Import default specifications
    from .defaults import (
        get_standard_domain_specs,
        get_standard_fuzzy_sets_specs,
        get_standard_rules_spec,
    )

    # When running the test harness, just run for two generations
    if os.environ.get(test_env_var, False) == "True":
        generations = 2
        pop_size = 5
    else:
        generations = 50
        pop_size = 20

    # Get specifications
    domain_specs = get_standard_domain_specs()
    fuzzy_set_specs = get_standard_fuzzy_sets_specs()
    initial_rule_specs = get_standard_rules_spec()

    # Define domain configuration
    input_domain_names = ["position", "velocity", "angle", "angular_velocity"]
    output_domain_name = "action"

    optimize_fuzzy_rules(
        domain_specs=domain_specs,
        fuzzy_set_specs=fuzzy_set_specs,
        input_domain_names=input_domain_names,
        output_domain_name=output_domain_name,
        pop_size=pop_size,
        generations=generations,
        num_episodes=3,
        max_steps=500,
        mutation_rate=0.15,
        use_initial_seed=False,  # Set to False for random initialization
        initial_rule_specs=initial_rule_specs,
        default_output_name="nothing",
        output_file="optimized_rules.csv",
        yaml_file="optimized_rules.yaml",
        use_parallel=True,  # Now safe for parallel execution
    )
