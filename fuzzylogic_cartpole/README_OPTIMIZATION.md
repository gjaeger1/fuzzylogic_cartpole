# Evolutionary Optimization of Fuzzy Rule Base

This module provides tools to optimize the fuzzy rule base for the CartPole controller using evolutionary algorithms with the LEAP library.

## Overview

The fuzzy controller has:
- **4 input domains**: position, velocity, angle, angular_velocity (each with 3 fuzzy sets: negative, zero, positive)
- **1 output domain**: action (with 5 fuzzy sets: strong_left, left, nothing, right, strong_right)
- **81 total rules**: One for each combination of input fuzzy sets (3×3×3×3)

The evolutionary algorithm optimizes which output fuzzy set (consequent) to use for each rule.

## Encoding

Each individual in the EA is represented as a vector of 81 integers, where:
- Each element corresponds to one specific combination of input fuzzy sets
- The value (0-4) encodes which output fuzzy set to use:
  - 0: `strong_left`
  - 1: `left`
  - 2: `nothing`
  - 3: `right`
  - 4: `strong_right`

## Installation

Make sure you have the required dependencies:

```bash
pip install leap-ec dask distributed gymnasium matplotlib pyyaml
```

## Usage

### 1. Optimize Rules with Evolutionary Algorithm

```bash
python -m fuzzylogic_cartpole.optimize_rules_ea
```

This will:
- Initialize a population (with the standard rule base as a seed)
- Run the EA with parallel fitness evaluation using dask
- Save the best genome to `optimized_rules.csv`
- Display live fitness plots

**Key Parameters** (edit in `optimize_rules_ea.py`):
- `pop_size`: Population size (default: 20)
- `generations`: Number of generations (default: 50)
- `num_episodes`: Episodes per fitness evaluation (default: 3)
- `max_steps`: Maximum steps per episode (default: 500)
- `mutation_rate`: Mutation probability per gene (default: 0.15)
- `use_parallel`: Enable parallel evaluation with dask (default: True)

### 2. Test and Save Optimized Rules

After optimization, decode and test the best rules:

```bash
python -m fuzzylogic_cartpole.test_optimized_rules --csv optimized_rules.csv --yaml optimized_rules.yaml
```

This will:
- Load the best genome from the CSV file
- Test it in the CartPole environment (10 episodes)
- Save the rules to a YAML file for future use

**Options**:
- `--csv FILE`: Input CSV file from EA (default: optimized_rules.csv)
- `--yaml FILE`: Output YAML file (default: optimized_rules.yaml)
- `--no-test`: Skip testing the controller
- `--render`: Render the test episodes visually

### 3. Use Optimized Rules in Your Code

Load and use the optimized controller:

```python
from fuzzylogic_cartpole.rule_base_generation import generate_controller_from_file

# Load controller from YAML file
controller = generate_controller_from_file("optimized_rules.yaml")

# Use in environment
import gymnasium as gym
env = gym.make("CartPole-v1")
observation, info = env.reset()

action = controller.get_action(observation)
```

## Implementation Details

### `optimize_rules_ea.py`

Main evolutionary optimization script with:

- **`get_input_combinations()`**: Generates all 81 input combinations
- **`decode_genome_to_rules()`**: Converts integer genome to rule specifications
- **`genome_to_controller()`**: Creates a FuzzyCartPoleController from genome
- **`create_fitness_function()`**: Evaluates controller performance in CartPole
- **`optimize_fuzzy_rules()`**: Main EA loop with LEAP

Uses LEAP's distributed evaluation features:
- `DistributedIndividual`: Tracks evaluation metadata
- `synchronous.eval_pool()`: Parallel fitness evaluation across CPU cores
- Dask Client: Manages parallel workers

### `test_optimized_rules.py`

Utility functions for working with optimized rules:

- **`decode_genome_from_csv()`**: Load genome from EA output
- **`genome_to_rule_specs()`**: Convert to rule specifications
- **`test_controller()`**: Evaluate controller performance
- **`save_optimized_rules_to_yaml()`**: Save to reusable YAML format

## Customization

### Modify EA Parameters

Edit the `optimize_fuzzy_rules()` call in the `if __name__ == "__main__"` block:

```python
optimize_fuzzy_rules(
    pop_size=30,              # Larger population
    generations=100,          # More generations
    num_episodes=5,           # More robust fitness evaluation
    max_steps=1000,           # Longer episodes
    mutation_rate=0.1,        # Lower mutation rate
    use_standard_seed=True,   # Start from standard rules
    use_parallel=True,        # Parallel evaluation
)
```

### Change EA Operators

In `optimize_rules_ea.py`, modify the pipeline:

```python
pipeline = [
    ops.tournament_selection,     # Selection operator
    ops.clone,
    mutate_randint(...),          # Mutation operator
    ops.UniformCrossover(...),    # Crossover operator
    # ... evaluation and probes
]
```

### Modify Fitness Function

Edit `create_fitness_function()` to change how controllers are evaluated:

```python
def create_fitness_function(domains, num_episodes=5, max_steps=500):
    def fitness_function(genome):
        # Custom fitness calculation
        # e.g., penalize for using too many different actions
        # or reward for stability
        pass
    return fitness_function
```

## Parallel Execution

### Local Machine (Default)

The script automatically uses all available CPU cores via dask:

```python
with Client() as client:
    # Uses all cores on local machine
    run_evolution(client)
```

### Cluster/Supercomputer

For cluster execution:

1. Start dask scheduler and workers:
```bash
dask-scheduler  # On head node
dask-worker tcp://scheduler-address:8786  # On each worker node
```

2. Modify the Client initialization:
```python
client = Client(scheduler_file='scheduler.json')
```

## Tips

1. **Start with standard rules**: Use `use_standard_seed=True` to initialize one individual with the working standard rule base
2. **Balance exploration/exploitation**: Adjust `mutation_rate` and population size
3. **Robust fitness**: Increase `num_episodes` for more stable fitness estimates
4. **Monitor progress**: Watch the dask dashboard for parallel execution stats
5. **Save checkpoints**: The CSV file is updated each generation with the best genome

## Example Output

```
Setting up fuzzy domains and sets...
Number of rules to optimize: 81
Starting evolutionary optimization...
Population size: 20
Generations: 50
Parallel evaluation: True
Starting dask client for parallel fitness evaluations...
Dask dashboard available at: http://127.0.0.1:8787/status
Number of workers: 8

generation, pop_size, min_fitness, max_fitness, ...
0, 20, 42.3, 387.6, ...
1, 20, 98.5, 412.8, ...
...
50, 20, 345.2, 498.1, ...

Optimization complete! Best genomes saved to optimized_rules.csv
```

## Troubleshooting

**ImportError: No module named 'leap_ec'**
- Install LEAP: `pip install leap-ec`

**Dask workers not starting**
- Check firewall settings
- Use `use_parallel=False` for sequential evaluation

**Low fitness scores**
- Increase `num_episodes` for more stable evaluation
- Try different mutation rates
- Increase population size or generations

**Memory issues with parallel evaluation**
- Reduce `pop_size`
- Use fewer dask workers
- Reduce `num_episodes` or `max_steps`
