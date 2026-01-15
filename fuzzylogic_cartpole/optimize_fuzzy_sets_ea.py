"""
Evolutionary Algorithm for Optimizing Fuzzy Set Parameters
Takes a fixed rule base and optimizes the membership function parameters.
Uses LEAP library with parallel evaluation via dask.
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
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from matplotlib import pyplot as plt

from .controller import FuzzyCartPoleController
from .rule_base_generation import (
    generate_domains,
    generate_fuzzy_sets,
    generate_rule_base,
    load_specification,
    save_specification,
)

##############################
# Encoding/Decoding Functions
##############################


def get_fuzzy_set_bounds(domain_specs, fuzzy_set_specs):
    """Calculate bounds for fuzzy set parameters.

    Args:
        domain_specs: List of domain specifications
        fuzzy_set_specs: List of fuzzy set specifications

    Returns:
        list: List of (min, max) tuples for each parameter in the genome
    """
    # Create domain lookup
    domain_dict = {spec["name"]: spec for spec in domain_specs}

    bounds = []

    for fs_spec in fuzzy_set_specs:
        domain_name = fs_spec["domain"]
        domain_spec = domain_dict[domain_name]

        domain_min = domain_spec["min"]
        domain_max = domain_spec["max"]

        # Add bounds for param1 and param2
        # Both parameters should be within the domain range
        bounds.append((domain_min, domain_max))
        bounds.append((domain_min, domain_max))

    return bounds


def decode_genome_to_fuzzy_sets(genome, fuzzy_set_specs_template):
    """Convert real-valued genome to fuzzy set specifications.

    Args:
        genome: Array of real values (2 per fuzzy set: param1, param2)
        fuzzy_set_specs_template: Template fuzzy set specs with domain, name, function

    Returns:
        list: Fuzzy set specifications with updated parameters
    """
    fuzzy_set_specs = []

    # Minimum epsilon to ensure param1 < param2
    EPSILON = 0.01

    for idx, template_spec in enumerate(fuzzy_set_specs_template):
        # Each fuzzy set uses 2 genome positions
        param1_idx = idx * 2
        param2_idx = idx * 2 + 1

        param1 = float(genome[param1_idx])
        param2 = float(genome[param2_idx])

        # Ensure proper ordering: param1 < param2 for all function types
        if param1 > param2:
            param1, param2 = param2, param1

        # Ensure minimum separation between parameters
        if param2 - param1 < EPSILON:
            # Adjust param2 to be at least EPSILON away from param1
            param2 = param1 + EPSILON

        # Create new fuzzy set spec with decoded parameters
        fuzzy_set_spec = {
            "domain": template_spec["domain"],
            "name": template_spec["name"],
            "function": template_spec["function"],
            "param1": param1,
            "param2": param2,
        }

        fuzzy_set_specs.append(fuzzy_set_spec)

    return fuzzy_set_specs


def genome_to_controller(
    genome,
    domain_specs,
    fuzzy_set_specs_template,
    rule_specs,
    default_outputs=None,
    verbose=False,
):
    """Convert genome to a FuzzyCartPoleController.

    Args:
        genome: Array of real values encoding fuzzy set parameters
        domain_specs: Domain specifications
        fuzzy_set_specs_template: Template fuzzy set specs (domain, name, function)
        rule_specs: Fixed rule specifications
        default_outputs: Default outputs for rules (if any)
        verbose: If True, controller will print debug information

    Returns:
        FuzzyCartPoleController: Controller with optimized fuzzy sets
    """
    # Decode genome to fuzzy set specifications
    fuzzy_set_specs = decode_genome_to_fuzzy_sets(genome, fuzzy_set_specs_template)

    # Generate domains
    domains = generate_domains(domain_specs)

    # Generate fuzzy sets with optimized parameters
    domains = generate_fuzzy_sets(domains, fuzzy_set_specs)

    # Generate rule base (fixed)
    rules = generate_rule_base(list(domains), rule_specs, default_outputs)

    # Create controller
    controller = FuzzyCartPoleController(domains, rules, verbose=verbose)

    return controller


##############################
# Fitness Evaluation
##############################


def evaluate_controller(controller, episodes=5, max_steps=500, render=False):
    """Evaluate a controller in the CartPole environment.

    Args:
        controller: FuzzyCartPoleController instance
        episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        float: Average total reward across episodes
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    total_rewards = []

    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Get action from fuzzy controller
            action = controller.get_action(observation)

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward - (
                abs(observation[0]) + abs(observation[2])
            )  # reduce by error on position and angle

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    env.close()

    # Return average reward
    avg_reward = np.mean(total_rewards)

    return avg_reward


def create_fitness_function(
    domain_specs,
    fuzzy_set_specs_template,
    rule_specs,
    default_outputs=None,
    episodes=5,
    max_steps=500,
):
    """Create a fitness function for the EA.

    Args:
        domain_specs: Domain specifications
        fuzzy_set_specs_template: Template fuzzy set specs
        rule_specs: Fixed rule specifications
        default_outputs: Default outputs for rules (if any)
        episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode

    Returns:
        function: Fitness function that takes a genome and returns fitness
    """

    def fitness_function(genome):
        """Evaluate genome fitness."""
        try:
            # Create controller from genome
            controller = genome_to_controller(
                genome,
                domain_specs,
                fuzzy_set_specs_template,
                rule_specs,
                default_outputs,
                verbose=False,
            )

            # Evaluate controller
            fitness = evaluate_controller(
                controller, episodes=episodes, max_steps=max_steps, render=False
            )

            return fitness

        except Exception as e:
            # Return poor fitness if there's an error
            print(f"Error evaluating genome: {e}", file=sys.stderr)
            return 0.0

    return fitness_function


##############################
# Probes for Monitoring
##############################


def build_probes(genomes_file=None):
    """Build probes for monitoring EA progress.

    Args:
        genomes_file: Optional file stream to write genome data

    Returns:
        list: List of probe functions
    """
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

    # Plot fitness curve (commented out to avoid display issues in headless environments)
    # plt.figure()
    # plt.ylabel("Fitness")
    # plt.xlabel("Generations")
    # plt.title("Best-of-Generation Fitness")
    # probes.append(
    #     probe.FitnessPlotProbe(ylim=(0, 500), xlim=(0, 1), modulo=1, ax=plt.gca())
    # )

    return probes


##############################
# Genome Creation
##############################


def create_initial_genome_from_specs(fuzzy_set_specs):
    """Create an initial genome from existing fuzzy set specifications.

    Args:
        fuzzy_set_specs: List of fuzzy set specifications with param1 and param2

    Returns:
        numpy.ndarray: Genome encoding the fuzzy set parameters
    """
    genome = []

    for spec in fuzzy_set_specs:
        genome.append(spec["param1"])
        genome.append(spec["param2"])

    return np.array(genome, dtype=float)


##############################
# Main EA Function
##############################


def optimize_fuzzy_sets(
    rule_base_file,
    output_file="optimized_fuzzy_sets.yaml",
    use_initial_seed=True,
    pop_size=50,
    generations=100,
    mutation_std=0.1,
    episodes_per_eval=5,
    max_steps=500,
    use_parallel=True,
    n_workers=4,
    log_file=None,
):
    """Optimize fuzzy set parameters using an evolutionary algorithm.

    Args:
        rule_base_file: Path to YAML file with rule base specification
        output_file: Path to save the optimized specification
        use_initial_seed: If True, seed population with parameters from rule_base_file
        pop_size: Population size
        generations: Number of generations
        mutation_std: Standard deviation for Gaussian mutation
        episodes_per_eval: Number of episodes per fitness evaluation
        max_steps: Maximum steps per episode
        use_parallel: Whether to use parallel evaluation with dask
        n_workers: Number of dask workers if use_parallel=True
        log_file: Optional file path for logging (default: stdout)

    Returns:
        numpy.ndarray: Best genome found
    """

    # Load rule base specification
    print(f"Loading rule base from {rule_base_file}...")
    domain_specs, fuzzy_set_specs, rule_specs, default_outputs = load_specification(
        rule_base_file
    )

    # Create template (without param1/param2) for fuzzy sets
    fuzzy_set_specs_template = [
        {
            "domain": spec["domain"],
            "name": spec["name"],
            "function": spec["function"],
        }
        for spec in fuzzy_set_specs
    ]

    # Calculate genome bounds
    bounds = get_fuzzy_set_bounds(domain_specs, fuzzy_set_specs)
    genome_length = len(fuzzy_set_specs_template) * 2  # 2 params per fuzzy set

    print(
        f"Genome length: {genome_length} (2 parameters × {len(fuzzy_set_specs_template)} fuzzy sets)"
    )
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Mutation std: {mutation_std}")

    # Create fitness function
    fitness_func = create_fitness_function(
        domain_specs,
        fuzzy_set_specs_template,
        rule_specs,
        default_outputs,
        episodes=episodes_per_eval,
        max_steps=max_steps,
    )

    # Wrap in FunctionProblem for LEAP
    problem = FunctionProblem(fitness_func, maximize=True)

    # Setup logging
    genomes_stream = None
    if log_file:
        genomes_stream = open(log_file, "w")

    probes_list = build_probes(genomes_stream)

    # Track best individual
    best_individual = {"genome": None, "fitness": -np.inf}

    def track_best(population):
        """Track best individual across generations."""
        for ind in population:
            if ind.fitness > best_individual["fitness"]:
                best_individual["fitness"] = ind.fitness
                best_individual["genome"] = np.copy(ind.genome)
        return population

    # Evolution function
    def run_evolution():
        """Run the evolutionary algorithm."""

        # Custom initializer for seeding with initial genome
        if use_initial_seed:
            # Create initial genome from provided specs
            initial_genome = create_initial_genome_from_specs(fuzzy_set_specs)

            # Create random genome generator
            random_genome_generator = create_real_vector(bounds)

            # Create iterator that yields initial genome first, then random
            def init_func():
                yield initial_genome
                while True:
                    yield random_genome_generator()

            init_func_iter = init_func()
            initialize = lambda: next(init_func_iter)
        else:
            # Fully random initialization
            initialize = create_real_vector(bounds)

        # Build pipeline
        pipeline = [
            ops.tournament_selection,
            ops.clone,
            mutate_gaussian(
                std=mutation_std,
                bounds=np.array(bounds),
                expected_num_mutations="isotropic",
            ),
        ]

        # Add evaluation operator
        if use_parallel:
            # Parallel evaluation already returns a list
            pipeline.append(synchronous.eval_pool(client=client, size=pop_size))
        else:
            # Sequential evaluation returns a generator, need to convert to list
            pipeline.append(ops.evaluate)
            pipeline.append(ops.pool(size=pop_size))

        # Add probes and tracking
        pipeline.extend(probes_list)
        pipeline.append(track_best)

        # Run EA
        final_pop = generational_ea(
            max_generations=generations,
            pop_size=pop_size,
            problem=problem,
            representation=Representation(
                individual_cls=DistributedIndividual if use_parallel else Individual,
                initialize=initialize,
            ),
            pipeline=pipeline,
        )

        return final_pop

    # Run evolution
    if use_parallel:
        print(f"Starting dask client with {n_workers} workers...")
        client = Client(n_workers=n_workers, threads_per_worker=1)
        print(f"Dask dashboard: {client.dashboard_link}")

    try:
        print("Starting evolution...")
        final_population = run_evolution()

        print("\nEvolution complete!")
        print(f"Best fitness: {best_individual['fitness']}")

    finally:
        if use_parallel:
            client.close()

        if genomes_stream is not None:
            genomes_stream.close()

    # Save best specification to YAML
    print(f"\nSaving best specification to {output_file}...")

    # Decode best genome to fuzzy set specs
    best_fuzzy_set_specs = decode_genome_to_fuzzy_sets(
        best_individual["genome"], fuzzy_set_specs_template
    )

    # Save complete specification
    save_specification(
        domain_specs,
        best_fuzzy_set_specs,
        rule_specs,
        output_file,
        default_outputs,
    )

    print(f"Specification saved to {output_file}")

    return best_individual["genome"]


##############################
# Main Entry Point
##############################


if __name__ == "__main__":
    # Example usage
    from .defaults import (
        get_standard_domain_specs,
        get_standard_fuzzy_sets_specs,
        get_standard_rules_spec,
    )

    print("=" * 80)
    print("FUZZY SET PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\nThis will optimize fuzzy set membership function parameters")
    print("while keeping the rule base fixed.\n")

    # First, create a rule base file if it doesn't exist
    rule_base_file = "optimized_rules_working.yaml"

    if not os.path.exists(rule_base_file):
        print(f"Creating initial rule base file: {rule_base_file}")
        domain_specs = get_standard_domain_specs()
        fuzzy_set_specs = get_standard_fuzzy_sets_specs()
        rule_specs = get_standard_rules_spec()
        default_outputs = {"action": "nothing"}

        save_specification(
            domain_specs,
            fuzzy_set_specs,
            rule_specs,
            rule_base_file,
            default_outputs,
        )
        print(f"✓ Rule base file created\n")
    else:
        print(f"Using existing rule base file: {rule_base_file}\n")

    # Optimize fuzzy sets
    print("Starting fuzzy set optimization...")
    print("-" * 80)
    best_genome = optimize_fuzzy_sets(
        rule_base_file=rule_base_file,
        output_file="optimized_fuzzy_sets.yaml",
        use_initial_seed=True,  # Start from standard fuzzy sets
        pop_size=20,
        generations=50,
        mutation_std=0.01,
        episodes_per_eval=3,
        max_steps=500,
        use_parallel=True,  # Set to True for parallel evaluation
        n_workers=4,
        log_file="fuzzy_sets_ea_log.csv",
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - optimized_fuzzy_sets.yaml (controller specification)")
    print(f"  - fuzzy_sets_ea_log.csv (optimization log)")
    print(f"\nBest fitness achieved: {best_genome}")
    print("\nTo use the optimized controller:")
    print(
        "  >>> from fuzzylogic_cartpole.rule_base_generation import generate_controller_from_file"
    )
    print(
        "  >>> controller = generate_controller_from_file('optimized_fuzzy_sets.yaml')"
    )
    print("  >>> action = controller.get_action(observation)")
