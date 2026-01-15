"""
Validation test script to verify EA optimization setup is correct.
Tests encoding, decoding, and fitness evaluation without running full EA.
"""

import numpy as np

from .controller import FuzzyCartPoleController
from .optimize_rules_ea import (
    create_fitness_function,
    create_initial_population_from_standard,
    decode_genome_to_rules,
    genome_to_controller,
    get_input_combinations,
    get_output_fuzzy_sets,
)
from .rule_base_generation import (
    generate_rule_base,
    get_standard_domains,
    get_standard_fuzzy_sets,
)


def test_basic_setup():
    """Test 1: Verify fuzzy logic system setup."""
    print("Test 1: Fuzzy logic system setup...")

    domains = get_standard_domains()
    assert len(domains) == 5, "Should have 5 domains"

    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )

    # Check each domain has expected fuzzy sets
    assert hasattr(position, "negative"), "Position should have 'negative' set"
    assert hasattr(position, "zero"), "Position should have 'zero' set"
    assert hasattr(position, "positive"), "Position should have 'positive' set"

    assert hasattr(action, "strong_left"), "Action should have 'strong_left' set"
    assert hasattr(action, "nothing"), "Action should have 'nothing' set"
    assert hasattr(action, "strong_right"), "Action should have 'strong_right' set"

    print("   ✓ All domains and fuzzy sets created successfully")


def test_input_combinations():
    """Test 2: Verify input combination generation."""
    print("\nTest 2: Input combination generation...")

    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )

    combinations = get_input_combinations(position, velocity, angle, angular_velocity)

    assert len(combinations) == 81, (
        f"Should have 81 combinations, got {len(combinations)}"
    )

    # Check first and last combinations
    first = combinations[0]
    last = combinations[-1]

    assert len(first) == 4, "Each combination should have 4 elements"
    assert len(last) == 4, "Each combination should have 4 elements"

    print(f"   ✓ Generated {len(combinations)} input combinations")
    print(f"   ✓ First combination: {[s.name for s in first]}")
    print(f"   ✓ Last combination: {[s.name for s in last]}")


def test_output_sets():
    """Test 3: Verify output fuzzy sets."""
    print("\nTest 3: Output fuzzy set ordering...")

    output_sets = get_output_fuzzy_sets(None)

    assert len(output_sets) == 5, f"Should have 5 output sets, got {len(output_sets)}"
    assert output_sets[0] == "strong_left", "Index 0 should be strong_left"
    assert output_sets[1] == "left", "Index 1 should be left"
    assert output_sets[2] == "nothing", "Index 2 should be nothing"
    assert output_sets[3] == "right", "Index 3 should be right"
    assert output_sets[4] == "strong_right", "Index 4 should be strong_right"

    print(f"   ✓ Output sets: {output_sets}")


def test_genome_encoding():
    """Test 4: Verify genome encoding and decoding."""
    print("\nTest 4: Genome encoding/decoding...")

    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    full_domains = (position, velocity, angle, angular_velocity, action)

    # Create a simple test genome (all "nothing" actions)
    test_genome = [2] * 81  # 2 = "nothing"

    # Decode to rules
    rule_specs = decode_genome_to_rules(test_genome, full_domains)

    assert len(rule_specs) == 81, f"Should have 81 rules, got {len(rule_specs)}"

    # Verify all rules use "nothing"
    for spec in rule_specs:
        assert spec["output"] == "nothing", "All rules should output 'nothing'"
        assert spec["output_domain"] == "action", "Output domain should be 'action'"
        assert len(spec["inputs"]) == 4, "Each rule should have 4 inputs"

    print("   ✓ Decoded 81 rules from genome")
    print(f"   ✓ First rule inputs: {rule_specs[0]['inputs']}")
    print(f"   ✓ First rule output: {rule_specs[0]['output']}")


def test_controller_creation():
    """Test 5: Verify controller can be created from genome."""
    print("\nTest 5: Controller creation from genome...")

    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    full_domains = (position, velocity, angle, angular_velocity, action)

    # Create random genome
    random_genome = np.random.randint(0, 5, size=81).tolist()

    # Convert to controller
    controller = genome_to_controller(random_genome, full_domains, verbose=False)

    assert isinstance(controller, FuzzyCartPoleController), (
        "Should create FuzzyCartPoleController"
    )
    assert controller.domains is not None, "Controller should have domains"
    assert controller.rules is not None, "Controller should have rules"
    assert controller.verbose == False, "Controller should be in silent mode"

    # Test getting an action
    test_observation = [0.0, 0.0, 0.0, 0.0]  # Centered state
    action = controller.get_action(test_observation)

    assert action in [0, 1], f"Action should be 0 or 1, got {action}"

    print("   ✓ Controller created successfully")
    print(f"   ✓ Test action for centered state: {action}")


def test_standard_genome():
    """Test 6: Verify standard genome creation."""
    print("\nTest 6: Standard genome creation...")

    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    full_domains = (position, velocity, angle, angular_velocity, action)

    # Create standard genome
    standard_genome = create_initial_population_from_standard(
        full_domains, pop_size=10, num_rules=81
    )

    assert len(standard_genome) == 81, (
        f"Standard genome should have 81 rules, got {len(standard_genome)}"
    )
    assert all(0 <= g <= 4 for g in standard_genome), "All genes should be in range 0-4"

    # Check it's not all the same
    unique_values = len(set(standard_genome))
    print(f"   ✓ Standard genome created with {unique_values} unique action types")

    # Count action types
    action_names = ["strong_left", "left", "nothing", "right", "strong_right"]
    counts = {name: standard_genome.count(i) for i, name in enumerate(action_names)}
    print(f"   ✓ Action distribution: {counts}")


def test_fitness_function():
    """Test 7: Verify fitness evaluation works."""
    print("\nTest 7: Fitness function evaluation...")

    domains = get_standard_domains()
    position, velocity, angle, angular_velocity, action = get_standard_fuzzy_sets(
        *domains
    )
    full_domains = (position, velocity, angle, angular_velocity, action)

    # Create fitness function (with minimal episodes for speed)
    fitness_func = create_fitness_function(full_domains, num_episodes=1, max_steps=100)

    # Create a simple genome
    test_genome = [2] * 81  # All "nothing"

    print("   Evaluating genome (this may take a few seconds)...")
    fitness = fitness_func(test_genome)

    assert isinstance(fitness, (int, float)), (
        f"Fitness should be numeric, got {type(fitness)}"
    )
    assert fitness >= 0, f"Fitness should be non-negative, got {fitness}"

    print(f"   ✓ Fitness evaluation completed: {fitness:.2f}")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("EA OPTIMIZATION SETUP VALIDATION")
    print("=" * 70)

    try:
        test_basic_setup()
        test_input_combinations()
        test_output_sets()
        test_genome_encoding()
        test_controller_creation()
        test_standard_genome()
        test_fitness_function()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nThe EA optimization setup is working correctly!")
        print("You can now run the full optimization with:")
        print("  python -m fuzzylogic_cartpole.optimize_rules_ea")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
