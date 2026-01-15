def get_standard_domain_specs():
    # Domain specifications
    domain_specs = [
        {"name": "position", "min": -4.8, "max": 4.8, "res": 0.01},
        {"name": "velocity", "min": -4.0, "max": 4.0, "res": 0.01},
        {"name": "angle", "min": -0.5, "max": 0.5, "res": 0.01},
        {"name": "angular_velocity", "min": -4.0, "max": 4.0, "res": 0.01},
        {"name": "action", "min": 0.0, "max": 1.0, "res": 0.01},
    ]

    return domain_specs


def get_standard_fuzzy_sets_specs():
    # Define specifications for all fuzzy sets
    specifications = [
        # Fuzzy sets for position
        {
            "domain": "position",
            "name": "negative",
            "function": "S",
            "param1": -1.0,
            "param2": 0,
        },
        {
            "domain": "position",
            "name": "zero",
            "function": "triangular",
            "param1": -0.5,
            "param2": 0.5,
        },
        {
            "domain": "position",
            "name": "positive",
            "function": "R",
            "param1": 0,
            "param2": 1.0,
        },
        # Fuzzy sets for velocity
        {
            "domain": "velocity",
            "name": "negative",
            "function": "S",
            "param1": -4.0,
            "param2": -0.5,
        },
        {
            "domain": "velocity",
            "name": "zero",
            "function": "triangular",
            "param1": -1.0,
            "param2": 1.0,
        },
        {
            "domain": "velocity",
            "name": "positive",
            "function": "R",
            "param1": 0.5,
            "param2": 4.0,
        },
        # Fuzzy sets for angle
        {
            "domain": "angle",
            "name": "negative",
            "function": "S",
            "param1": -0.5,
            "param2": -0.05,
        },
        {
            "domain": "angle",
            "name": "zero",
            "function": "triangular",
            "param1": -0.1,
            "param2": 0.1,
        },
        {
            "domain": "angle",
            "name": "positive",
            "function": "R",
            "param1": 0.05,
            "param2": 0.5,
        },
        # Fuzzy sets for angular_velocity
        {
            "domain": "angular_velocity",
            "name": "negative",
            "function": "S",
            "param1": -4.0,
            "param2": -0.5,
        },
        {
            "domain": "angular_velocity",
            "name": "zero",
            "function": "triangular",
            "param1": -0.5,
            "param2": 0.5,
        },
        {
            "domain": "angular_velocity",
            "name": "positive",
            "function": "R",
            "param1": 0.5,
            "param2": 4.0,
        },
        # Fuzzy sets for action
        {
            "domain": "action",
            "name": "strong_left",
            "function": "S",
            "param1": 0.0,
            "param2": 0.25,
        },
        {
            "domain": "action",
            "name": "left",
            "function": "triangular",
            "param1": 0.1,
            "param2": 0.5,
        },
        {
            "domain": "action",
            "name": "nothing",
            "function": "triangular",
            "param1": 0.425,
            "param2": 0.575,
        },
        {
            "domain": "action",
            "name": "right",
            "function": "triangular",
            "param1": 0.5,
            "param2": 0.9,
        },
        {
            "domain": "action",
            "name": "strong_right",
            "function": "R",
            "param1": 0.75,
            "param2": 1.0,
        },
    ]

    return specifications


def get_standard_rules_spec():
    # Define specifications for rules that differ from the default (action.nothing)
    specifications = [
        # Position: negative, Velocity: negative
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: negative, Velocity: zero
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.zero",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: negative, Velocity: positive
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.positive",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.positive",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.positive",
                "angle.negative",
                "angular_velocity.positive",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.negative",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        # Position: zero, Velocity: negative
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.positive",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: zero, Velocity: zero
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.zero",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: zero, Velocity: positive
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.positive",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.zero",
                "velocity.positive",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: positive, Velocity: negative
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.negative",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.negative",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: positive, Velocity: zero
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.zero",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        # Position: positive, Velocity: positive
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.negative",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.negative",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.negative",
            ],
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.zero",
                "angular_velocity.positive",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.positive",
                "angular_velocity.zero",
            ],
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": [
                "position.positive",
                "velocity.positive",
                "angle.positive",
                "angular_velocity.positive",
            ],
            "output": "strong_right",
        },
    ]

    return specifications


def get_standard_specifications():
    """Get all standard specifications for the CartPole controller

    Returns:
        tuple: (domain_specs, fuzzy_set_specs, rule_specs) that can be passed to
               generate_domains, generate_fuzzy_sets, and generate_rule_base
    """
    # Domain specifications
    domain_specs = get_standard_domain_specs()

    # Fuzzy set specifications
    fuzzy_set_specs = get_standard_fuzzy_sets_specs()

    # Rule specifications (using string-based format)
    rule_specs = get_standard_rules_spec()

    return domain_specs, fuzzy_set_specs, rule_specs
