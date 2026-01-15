"""
Fuzzy Logic Controller for CartPole Environment
"""

import numpy as np
from fuzzylogic.classes import Domain, Rule, Set
from fuzzylogic.functions import R, S, trapezoid, triangular


def generate_domains(specification):
    """Generate Domain object from specification."""
    # specification is a list of individual domain specifications.
    # Each domain specification defines the name, min-value, max-value and resulition
    # The function should return a tuple, that is, the same datastructure as returned by 'get_standard_domains'

    domains = []
    for spec in specification:
        name = spec["name"]
        min_value = spec["min"]
        max_value = spec["max"]
        resolution = spec.get("res", 0.01)  # Default resolution if not specified

        domain = Domain(name, min_value, max_value, res=resolution)
        domains.append(domain)

    return tuple(domains)


def generate_fuzzy_sets(domains, specifications):
    """Generate fuzzy sets on domains"""
    # Domains is a tuple of domains
    # specificaiton is a list of individual fuzzy specifications
    # each fuzzy specification defines the name of the fuzzy set, a function representing the membership (S, R, triangular) and two parameters of the function
    # This function then generates the fuzzy set on the domain and returns the extended domains as a tuple

    # Create a dictionary to easily access domains by name
    domain_dict = {domain._name: domain for domain in domains}

    # Process each fuzzy set specification
    for spec in specifications:
        domain_name = spec["domain"]
        set_name = spec["name"]
        func_type = spec["function"]
        param1 = spec["param1"]
        param2 = spec["param2"]

        # Get the corresponding domain
        domain = domain_dict[domain_name]

        # Create the fuzzy set based on the function type
        if func_type == "S":
            func = S(param1, param2)
        elif func_type == "R":
            func = R(param1, param2)
        elif func_type == "triangular":
            func = triangular(param1, param2)
        else:
            raise ValueError(f"Unknown function type: {func_type}")

        # Create a Set object with the function
        fuzzy_set = Set(func)
        setattr(domain, set_name, fuzzy_set)

    return domains


def generate_rule_base(
    input_domains, output_domains, specifications, default_outputs=None
):
    """Generate fuzzy rules"""
    # For each combination of possible fuzzy sets on the input_domains there need to be a rule specified in the specifications. Otherwise, if given, default outputs need to be assigned.
    # if multiple output domains are present, separate rule bases are generated. One for each output domain

    import itertools

    # Convert to lists if not already
    if not isinstance(input_domains, (list, tuple)):
        input_domains = [input_domains]
    if not isinstance(output_domains, (list, tuple)):
        output_domains = [output_domains]

    # Get all fuzzy sets for each input domain
    # Access the _sets dictionary directly from each Domain object
    input_fuzzy_sets = []
    for domain in input_domains:
        # Get all fuzzy sets defined on this domain from the _sets dictionary
        sets = list(domain._sets.values())
        input_fuzzy_sets.append(sets)

    # Generate all combinations of input fuzzy sets
    all_combinations = list(itertools.product(*input_fuzzy_sets))

    # Create rule bases for each output domain
    rule_bases = {}

    for output_domain in output_domains:
        output_name = output_domain._name
        rules_dict = {}

        # Process each combination
        for combo in all_combinations:
            # Check if this combination is in specifications
            rule_found = False

            for spec in specifications:
                # Check if this spec matches the current combination and output domain
                if spec.get("output_domain") == output_name:
                    # Match input combination
                    spec_inputs = spec.get("inputs")
                    if spec_inputs == combo or (
                        isinstance(spec_inputs, (list, tuple))
                        and tuple(spec_inputs) == combo
                    ):
                        # Found a matching rule
                        output_set_name = spec.get("output")
                        output_set = getattr(output_domain, output_set_name)
                        rules_dict[combo] = output_set
                        rule_found = True
                        break

            # If no rule found, use default if provided
            if not rule_found and default_outputs is not None:
                if output_name in default_outputs:
                    default_set_name = default_outputs[output_name]
                    default_set = getattr(output_domain, default_set_name)
                    rules_dict[combo] = default_set

        # Create Rule object for this output domain
        if rules_dict:
            rule_bases[output_name] = Rule(rules_dict)

    # Return single rule base if only one output domain, otherwise return dictionary
    if len(output_domains) == 1:
        return rule_bases[output_domains[0]._name]
    else:
        return rule_bases


#####################################
#
# Defaults
#
#####################################


def get_standard_domains():
    # Define input domains
    # Cart position: typically -2.4 to 2.4
    position = Domain("position", -4.8, 4.8, res=0.01)

    # Cart velocity: typically -inf to inf, but practically -2 to 2
    velocity = Domain("velocity", -4.0, 4.0, res=0.01)

    # Pole angle: typically -0.418 to 0.418 radians (~24 degrees)
    angle = Domain("angle", -0.5, 0.5, res=0.01)

    # Pole angular velocity: typically -inf to inf, but practically -2 to 2
    angular_velocity = Domain("angular_velocity", -4.0, 4.0, res=0.01)

    # Define output domain
    # Action: 0 (push left) or 1 (push right)
    # We'll use a continuous output and threshold at 0.5
    action = Domain("action", 0.0, 1.0, res=0.01)

    return position, velocity, angle, angular_velocity, action


def get_standard_fuzzy_sets(position, velocity, angle, angular_velocity, action):
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

    # Use generate_fuzzy_sets to create all fuzzy sets
    domains = (position, velocity, angle, angular_velocity, action)
    generate_fuzzy_sets(domains, specifications)

    return position, velocity, angle, angular_velocity, action


def get_standard_rules(position, velocity, angle, angular_velocity, action):
    # Define specifications for rules that differ from the default (action.nothing)
    specifications = [
        # Position: negative, Velocity: negative
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: negative, Velocity: zero
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: negative, Velocity: positive
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.positive,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.positive,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.negative,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        # Position: zero, Velocity: negative
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "strong_left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.positive,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: zero, Velocity: zero
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: zero, Velocity: positive
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.positive,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.positive,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.zero,
                velocity.positive,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: positive, Velocity: negative
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.negative,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: positive, Velocity: zero
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        # Position: positive, Velocity: positive
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.negative,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.negative,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.negative,
            ),
            "output": "left",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.positive,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.positive,
                angular_velocity.zero,
            ),
            "output": "right",
        },
        {
            "output_domain": "action",
            "inputs": (
                position.positive,
                velocity.positive,
                angle.positive,
                angular_velocity.positive,
            ),
            "output": "strong_right",
        },
    ]

    # Use generate_rule_base with action.nothing as default
    input_domains = [position, velocity, angle, angular_velocity]
    output_domains = [action]
    default_outputs = {"action": "nothing"}

    rules = generate_rule_base(
        input_domains, output_domains, specifications, default_outputs
    )

    return rules
