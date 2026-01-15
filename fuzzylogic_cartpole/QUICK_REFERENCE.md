# Quick Reference Guide - EA Optimization of Fuzzy Rules

## 📁 Files Created

### Main Implementation Files

#### 1. `optimize_rules_ea.py` ⭐ MAIN SCRIPT
**Purpose:** Full-featured evolutionary algorithm for optimizing fuzzy rule bases

**Key Functions:**
- `get_input_combinations()` - Generate all 81 input fuzzy set combinations
- `decode_genome_to_rules()` - Convert integer genome to rule specifications
- `genome_to_controller()` - Create FuzzyCartPoleController from genome
- `create_fitness_function()` - Build fitness evaluator for CartPole
- `create_initial_population_from_standard()` - Seed population with standard rules
- `optimize_fuzzy_rules()` - Main EA optimization loop

**Usage:**
```bash
python -m fuzzylogic_cartpole.optimize_rules_ea
```

**Features:**
- ✅ Parallel evaluation with dask
- ✅ Live fitness plots
- ✅ Progress tracking
- ✅ Best genome logging to CSV
- ✅ Standard rule seeding

---

#### 2. `test_optimized_rules.py`
**Purpose:** Load, test, and save optimized rules

**Key Functions:**
- `decode_genome_from_csv()` - Load best genome from EA output
- `genome_to_rule_specs()` - Convert genome to YAML format
- `test_controller()` - Evaluate controller in CartPole
- `save_optimized_rules_to_yaml()` - Save rules for reuse

**Usage:**
```bash
# Test and save optimized rules
python -m fuzzylogic_cartpole.test_optimized_rules \
    --csv optimized_rules.csv \
    --yaml optimized_rules.yaml \
    --render
```

**Features:**
- ✅ CSV to YAML conversion
- ✅ Performance testing (10 episodes)
- ✅ Statistical analysis
- ✅ Optional visualization

---

#### 3. `example_optimize_minimal.py`
**Purpose:** Simplified optimization for quick testing

**Usage:**
```bash
python -m fuzzylogic_cartpole.example_optimize_minimal
```

**Configuration:**
- Population: 10
- Generations: 10
- Episodes: 2
- Parallel: Off (default)
- Runtime: ~5-10 minutes

**Best for:**
- Quick validation
- Testing setup
- Learning the workflow

---

#### 4. `demo_ea_optimization.py`
**Purpose:** Educational demonstration with detailed explanations

**Usage:**
```python
from fuzzylogic_cartpole.demo_ea_optimization import demo
demo()
```

**Features:**
- ✅ Step-by-step walkthrough
- ✅ Inline documentation
- ✅ Minimal dependencies
- ✅ Shows core concepts

**Best for:**
- Understanding the approach
- Learning EA basics
- Code examples

---

#### 5. `test_ea_setup.py`
**Purpose:** Validation tests for EA setup

**Usage:**
```bash
python -m fuzzylogic_cartpole.test_ea_setup
```

**Tests:**
1. Fuzzy logic system setup
2. Input combination generation (81 rules)
3. Output fuzzy set ordering
4. Genome encoding/decoding
5. Controller creation
6. Standard genome initialization
7. Fitness evaluation

**Output:** Pass/fail for each component

---

### Documentation Files

#### 6. `README_OPTIMIZATION.md`
**Complete user guide** covering:
- Overview and architecture
- Installation instructions
- Usage examples
- Customization options
- Parallel execution
- Troubleshooting

#### 7. `SUMMARY_EA_OPTIMIZATION.md`
**Technical reference** including:
- Detailed architecture
- Encoding schemes
- Algorithm descriptions
- Expected results
- Best practices
- API reference

#### 8. `QUICK_REFERENCE.md` (this file)
**Quick lookup** for:
- File overview
- Common commands
- Code snippets
- Workflow guide

---

## 🚀 Quick Start Workflows

### Workflow 1: Test Setup (30 seconds)
```bash
# Verify everything works
python -m fuzzylogic_cartpole.test_ea_setup
```

### Workflow 2: Quick Demo (5 minutes)
```bash
# Run minimal optimization
python -m fuzzylogic_cartpole.example_optimize_minimal
```

### Workflow 3: Full Optimization (30-60 minutes)
```bash
# Run full EA with parallel evaluation
python -m fuzzylogic_cartpole.optimize_rules_ea

# Test and save results
python -m fuzzylogic_cartpole.test_optimized_rules \
    --csv optimized_rules.csv \
    --yaml optimized_rules.yaml
```

### Workflow 4: Use Optimized Rules
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

---

## 🧬 Genome Encoding Reference

### Structure
- **Type:** List of 81 integers
- **Range:** 0-4 (output fuzzy set index)
- **Order:** Lexicographic input combinations

### Output Encoding
```
0 = strong_left    (strong push left)
1 = left           (moderate push left)
2 = nothing        (neutral/no action)
3 = right          (moderate push right)
4 = strong_right   (strong push right)
```

### Example Genome
```python
genome = [2, 1, 0, 3, 4, 2, ...]  # 81 integers
         │  │  │  │  │  └─ Rule 5 → nothing
         │  │  │  │  └──── Rule 4 → strong_right
         │  │  │  └─────── Rule 3 → right
         │  │  └────────── Rule 2 → strong_left
         │  └───────────── Rule 1 → left
         └──────────────── Rule 0 → nothing
```

---

## ⚙️ Common Customizations

### Change EA Parameters
Edit in `optimize_rules_ea.py`:
```python
optimize_fuzzy_rules(
    pop_size=30,           # Default: 20
    generations=100,       # Default: 50
    num_episodes=5,        # Default: 3
    max_steps=1000,        # Default: 500
    mutation_rate=0.1,     # Default: 0.15
    use_parallel=True,     # Default: True
)
```

### Modify EA Pipeline
```python
pipeline = [
    ops.tournament_selection,      # Selection
    ops.clone,                      # Reproduction
    mutate_randint(...),            # Mutation
    ops.UniformCrossover(...),      # Crossover
    synchronous.eval_pool(...),     # Evaluation
    *build_probes(...)              # Monitoring
]
```

### Custom Fitness Function
```python
def create_fitness_function(domains, num_episodes, max_steps):
    def fitness_function(genome):
        controller = genome_to_controller(genome, domains)
        
        # Your custom evaluation here
        fitness = evaluate_custom(controller)
        
        return fitness
    return fitness_function
```

---

## 📊 Output Files

### `optimized_rules.csv`
Generated by: `optimize_rules_ea.py`
```csv
generation,individual,fitness,genome
0,0,245.3,"[2, 1, 0, 3, ...]"
1,0,298.7,"[2, 1, 1, 3, ...]"
...
```

### `optimized_rules.yaml`
Generated by: `test_optimized_rules.py`
```yaml
domains:
  - name: position
    min: -4.8
    max: 4.8
fuzzy_sets:
  - domain: position
    name: negative
    ...
rules:
  - output_domain: action
    inputs: [...]
    output: left
```

---

## 🐛 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| ImportError: leap_ec | `pip install leap-ec` |
| Low fitness scores | Increase `num_episodes` to 5+ |
| Slow evaluation | Set `use_parallel=True` |
| Memory issues | Reduce `pop_size` or `num_episodes` |
| No improvement | Try `use_standard_seed=True` |
| Dask not starting | Set `use_parallel=False` |

---

## 📦 Dependencies

```bash
pip install leap-ec dask distributed gymnasium numpy matplotlib pyyaml
```

Or from requirements:
```txt
leap-ec>=0.8.0
dask>=2023.0.0
distributed>=2023.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
fuzzylogic>=1.1.0
```

---

## 🎯 Expected Performance

### Standard Rules (Baseline)
- Fitness: 150-250
- Episodes: Variable length
- Success rate: ~50%

### Optimized Rules (After 50 generations)
- Fitness: 350-500
- Episodes: Often max length (500 steps)
- Success rate: 80-100%
- Improvement: 50-200%

---

## 💡 Tips & Best Practices

1. **Start with validation:** Run `test_ea_setup.py` first
2. **Use standard seed:** Better starting point with `use_standard_seed=True`
3. **Monitor progress:** Watch dask dashboard at http://localhost:8787
4. **Save checkpoints:** CSV updated every generation
5. **Test thoroughly:** Use 10+ episodes for final evaluation
6. **Adjust mutation:** Higher (0.2) for exploration, lower (0.1) for refinement
7. **Balance speed/accuracy:** More episodes = better fitness estimate but slower

---

## 🔗 Quick Links

- **LEAP Docs:** https://leap-ec.readthedocs.io/
- **Dask Docs:** https://docs.dask.org/
- **CartPole:** https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Fuzzy Logic:** https://github.com/amogorkon/fuzzylogic

---

## 📝 Cheat Sheet

### Import Statements
```python
from fuzzylogic_cartpole.optimize_rules_ea import (
    genome_to_controller,
    create_fitness_function,
)
from fuzzylogic_cartpole.rule_base_generation import (
    get_standard_domains,
    get_standard_fuzzy_sets,
    generate_controller_from_file,
)
```

### Common Operations
```python
# Setup domains
domains = get_standard_domains()
pos, vel, ang, ang_vel, act = get_standard_fuzzy_sets(*domains)
full_domains = (pos, vel, ang, ang_vel, act)

# Create controller from genome
genome = [2] * 81  # Example: all "nothing"
controller = genome_to_controller(genome, full_domains)

# Evaluate fitness
fitness_func = create_fitness_function(full_domains, 3, 500)
fitness = fitness_func(genome)

# Load from file
controller = generate_controller_from_file("rules.yaml")
```

---

**Last Updated:** 2024
**Version:** 1.0