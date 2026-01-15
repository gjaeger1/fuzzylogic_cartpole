# Evolutionary Algorithm Optimization for Fuzzy Rule Base - Summary

## Overview

This package provides a complete implementation for optimizing fuzzy logic controller rule bases using evolutionary algorithms (EA) with the LEAP library. The goal is to evolve better rule bases for the CartPole control problem by optimizing which output fuzzy sets (consequents) are assigned to each rule.

## Architecture

### 1. Fuzzy Logic System Structure

**Input Domains (4):**
- `position`: Cart position on track (-4.8 to 4.8)
- `velocity`: Cart velocity (-4.0 to 4.0)
- `angle`: Pole angle from vertical (-0.5 to 0.5 radians)
- `angular_velocity`: Pole angular velocity (-4.0 to 4.0)

**Input Fuzzy Sets (3 per domain):**
- `negative`: Left/negative side
- `zero`: Center/neutral
- `positive`: Right/positive side

**Output Domain (1):**
- `action`: Control action (0.0 to 1.0, defuzzified to discrete 0 or 1)

**Output Fuzzy Sets (5):**
- `strong_left` (index 0): Strong push left
- `left` (index 1): Moderate push left
- `nothing` (index 2): No action/neutral
- `right` (index 3): Moderate push right
- `strong_right` (index 4): Strong push right

**Total Rules:** 3 × 3 × 3 × 3 = **81 rules** (all combinations of input fuzzy sets)

### 2. Genome Encoding

Each individual in the EA is represented as:
- **Type:** Vector/list of integers
- **Length:** 81 (one element per rule)
- **Values:** 0-4 (index of output fuzzy set)
- **Example:** `[2, 1, 0, 3, 4, 2, ...]` means:
  - Rule 0 uses `nothing` (index 2)
  - Rule 1 uses `left` (index 1)
  - Rule 2 uses `strong_left` (index 0)
  - etc.

**Rule Ordering:**
Rules correspond to input combinations in lexicographic order:
```
Rule 0:  position.negative, velocity.negative, angle.negative, angular_velocity.negative
Rule 1:  position.negative, velocity.negative, angle.negative, angular_velocity.zero
Rule 2:  position.negative, velocity.negative, angle.negative, angular_velocity.positive
...
Rule 80: position.positive, velocity.positive, angle.positive, angular_velocity.positive
```

### 3. Decoding Process

**Genome → Controller Pipeline:**

1. **Genome:** `[int, int, ..., int]` (81 integers, 0-4)
2. **Decode to rule specs:** Convert each integer to corresponding output fuzzy set name
3. **Generate rule base:** Use `generate_rule_base()` to create fuzzy rules
4. **Create controller:** Instantiate `FuzzyCartPoleController` with domains and rules

**Key Function:** `genome_to_controller(genome, domains, verbose=False)`

### 4. Fitness Evaluation

**Objective:** Maximize average reward in CartPole-v1 environment

**Evaluation Process:**
1. Decode genome to controller
2. Run N episodes (typically 3-5)
3. Each episode: max 500 steps
4. Calculate average total reward
5. Return as fitness value

**Key Function:** `create_fitness_function(domains, num_episodes, max_steps)`

### 5. Evolutionary Algorithm

**Library:** LEAP (Lightweight Evolutionary Algorithms in Python)

**Operators:**
- **Selection:** Tournament selection
- **Reproduction:** Cloning
- **Mutation:** Random integer mutation (randomly change some genes to 0-4)
- **Crossover:** Uniform crossover (swap genes between parents)
- **Evaluation:** Fitness evaluation (sequential or parallel)

**Pipeline:**
```
Population → Selection → Clone → Mutate → Crossover → Evaluate → Next Generation
```

**Parameters (typical):**
- Population size: 10-30
- Generations: 50-100
- Mutation rate: 0.10-0.20 (10-20% of genes)
- Crossover probability: 0.1 (10% swap rate)

## Implementation Files

### Core Files

#### `optimize_rules_ea.py`
Main optimization script with full features:
- **Functions:**
  - `get_input_combinations()`: Generate all 81 input combinations
  - `get_output_fuzzy_sets()`: Return ordered output set names
  - `decode_genome_to_rules()`: Convert genome to rule specifications
  - `genome_to_controller()`: Convert genome to controller object
  - `create_fitness_function()`: Create fitness evaluation function
  - `create_initial_population_from_standard()`: Seed with standard rules
  - `optimize_fuzzy_rules()`: Main EA loop
- **Features:**
  - Parallel evaluation with dask
  - Progress monitoring with probes
  - Live fitness plots
  - Best genome logging
  - Standard rule seeding

#### `test_optimized_rules.py`
Utility for testing and saving results:
- **Functions:**
  - `decode_genome_from_csv()`: Load genome from EA output
  - `genome_to_rule_specs()`: Convert genome to YAML-compatible specs
  - `test_controller()`: Evaluate controller performance
  - `save_optimized_rules_to_yaml()`: Save to reusable format
- **Features:**
  - Load best genome from CSV
  - Test in CartPole environment
  - Save to YAML for later use
  - Performance statistics

#### `example_optimize_minimal.py`
Simplified example for quick testing:
- Reduced parameters (10 pop, 10 gen)
- Sequential evaluation (no dask)
- Minimal output
- Quick validation

#### `demo_ea_optimization.py`
Educational demonstration:
- Step-by-step explanation
- Inline comments
- Minimal dependencies
- Shows core concepts clearly

### Modified Files

#### `controller.py`
Added `verbose` parameter:
- `verbose=False`: Silent operation (for EA)
- `verbose=True`: Print debug info (for testing)

## Usage Guide

### Quick Start (5 generations, no parallelism)

```python
from fuzzylogic_cartpole.demo_ea_optimization import demo
demo()
```

### Minimal Optimization (10 generations)

```bash
python -m fuzzylogic_cartpole.example_optimize_minimal
```

### Full Optimization (50+ generations, parallel)

```bash
python -m fuzzylogic_cartpole.optimize_rules_ea
```

### Test Optimized Rules

```bash
python -m fuzzylogic_cartpole.test_optimized_rules --csv optimized_rules.csv --yaml optimized_rules.yaml
```

### Use in Custom Code

```python
from fuzzylogic_cartpole.rule_base_generation import generate_controller_from_file

# Load optimized controller
controller = generate_controller_from_file("optimized_rules.yaml")

# Use in environment
import gymnasium as gym
env = gym.make("CartPole-v1")
observation, _ = env.reset()
action = controller.get_action(observation)
```

## Key Algorithms

### Genome to Controller

```python
def genome_to_controller(genome, domains):
    # 1. Get all 81 input combinations
    combinations = get_input_combinations(position, velocity, angle, angular_velocity)
    
    # 2. Output fuzzy set names
    outputs = ["strong_left", "left", "nothing", "right", "strong_right"]
    
    # 3. Build rule specifications
    rules = []
    for i, combo in enumerate(combinations):
        output_name = outputs[genome[i]]  # Decode integer to name
        rule = {
            "output_domain": "action",
            "inputs": [f"{domain}.{fset._name}" for domain, fset in zip(..., combo)],
            "output": output_name
        }
        rules.append(rule)
    
    # 4. Generate rule base
    rule_base = generate_rule_base(domains, rules)
    
    # 5. Create controller
    return FuzzyCartPoleController(domains, rule_base)
```

### Fitness Evaluation

```python
def evaluate_fitness(genome):
    # Decode genome to controller
    controller = genome_to_controller(genome, domains)
    
    # Evaluate over multiple episodes
    total_reward = 0
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = controller.get_action(observation)
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes
```

## Parallel Execution

### Local (Default)

Uses all CPU cores automatically:

```python
from distributed import Client

with Client() as client:
    # Dashboard at http://localhost:8787
    optimize_fuzzy_rules(use_parallel=True)
```

### Cluster/Supercomputer

```bash
# On head node
dask-scheduler

# On worker nodes
dask-worker tcp://scheduler-address:8786

# In Python
client = Client(scheduler_file='scheduler.json')
```

## Expected Results

### Standard Rules Baseline
- Fitness: ~150-250 (varies by random seed)
- Performance: Can balance pole for limited time

### After Optimization (50 generations)
- Fitness: ~350-500
- Performance: Often achieves maximum episode length
- Improvement: 50-200% over standard rules

### Example Output

```
Setting up fuzzy domains and sets...
Number of rules to optimize: 81
Starting evolutionary optimization...
Population size: 20
Generations: 50
Parallel evaluation: True

generation, pop_size, min_fitness, max_fitness, ...
0, 20, 42.3, 387.6, ...
10, 20, 198.5, 445.2, ...
25, 20, 287.3, 481.7, ...
50, 20, 345.2, 498.1, ...

Optimization complete! Best genomes saved to optimized_rules.csv
```

## Customization Options

### Change EA Parameters

```python
optimize_fuzzy_rules(
    pop_size=30,           # Larger population
    generations=100,       # More generations
    num_episodes=5,        # More robust evaluation
    max_steps=1000,        # Longer episodes
    mutation_rate=0.1,     # Less mutation
    use_standard_seed=True # Start from known good solution
)
```

### Modify EA Operators

```python
pipeline = [
    ops.tournament_selection(k=3),  # Larger tournament
    ops.clone,
    mutate_randint(...),
    ops.TwoPointCrossover(),        # Different crossover
    ...
]
```

### Custom Fitness Function

```python
def custom_fitness(genome):
    controller = genome_to_controller(genome, domains)
    
    # Penalize complex rules
    unique_actions = len(set(genome))
    complexity_penalty = -0.1 * unique_actions
    
    # Reward performance
    reward = evaluate_controller(controller, env)
    
    return reward + complexity_penalty
```

## Best Practices

1. **Start with standard seed:** Use `use_standard_seed=True` to initialize from known working rules
2. **Balance evaluation:** More episodes = more robust but slower
3. **Monitor dashboard:** Watch dask dashboard for bottlenecks
4. **Save checkpoints:** CSV file updated each generation
5. **Test thoroughly:** Evaluate final solution with many episodes
6. **Adjust mutation rate:** Higher for exploration, lower for refinement
7. **Use parallel execution:** Significant speedup on multi-core machines

## Troubleshooting

**Low fitness scores:**
- Increase `num_episodes` for more stable evaluation
- Try different mutation rates (0.05-0.25)
- Increase population size or generations
- Check if standard rules work (baseline validation)

**Slow evaluation:**
- Enable parallel execution (`use_parallel=True`)
- Reduce `num_episodes` or `max_steps`
- Use fewer probes
- Ensure `verbose=False` in controller

**Memory issues:**
- Reduce population size
- Limit dask workers
- Reduce episode length

**Unstable results:**
- Increase `num_episodes` for fitness evaluation
- Use larger population
- Lower mutation rate
- Add more generations

## Dependencies

```
leap-ec>=0.8.0
dask>=2023.0.0
distributed>=2023.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
fuzzylogic>=1.1.0
```

## References

- **LEAP Documentation:** https://leap-ec.readthedocs.io/
- **Dask Documentation:** https://docs.dask.org/
- **Gymnasium (CartPole):** https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Fuzzy Logic:** https://github.com/amogorkon/fuzzylogic

## License

Same as parent project.