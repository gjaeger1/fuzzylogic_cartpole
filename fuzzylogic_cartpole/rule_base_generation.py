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
            fuzzy_set = S(param1, param2)
        elif func_type == "R":
            fuzzy_set = R(param1, param2)
        elif func_type == "triangular":
            fuzzy_set = triangular(param1, param2)
        else:
            raise ValueError(f"Unknown function type: {func_type}")

        # Assign the fuzzy set to the domain
        setattr(domain, set_name, fuzzy_set)

    return domains


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
    rules = Rule(
        {
            # Complete rule base covering all 81 combinations (3^4)
            # Format: (position, velocity, angle, angular_velocity): action
            # Priority: Angle control > Angular velocity > Position > Velocity
            # ========== POSITION: NEGATIVE ==========
            # Position: negative, Velocity: negative
            (
                position.negative,
                velocity.negative,
                angle.negative,
                angular_velocity.negative,
            ): action.strong_left,
            (
                position.negative,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.negative,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.negative,
                velocity.negative,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.negative,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.negative,
                angle.zero,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.negative,
                velocity.negative,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.negative,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: negative, Velocity: zero
            (
                position.negative,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ): action.left,
            (
                position.negative,
                velocity.zero,
                angle.negative,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.zero,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ): action.left,
            (
                position.negative,
                velocity.zero,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.negative,
                velocity.zero,
                angle.positive,
                angular_velocity.zero,
            ): action.nothing,
            (
                position.negative,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: negative, Velocity: positive
            (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.negative,
            ): action.strong_left,
            (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.positive,
                angle.negative,
                angular_velocity.positive,
            ): action.left,
            (
                position.negative,
                velocity.positive,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.negative,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ): action.left,
            (
                position.negative,
                velocity.positive,
                angle.zero,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.negative,
                velocity.positive,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.negative,
                velocity.positive,
                angle.positive,
                angular_velocity.zero,
            ): action.nothing,
            (
                position.negative,
                velocity.positive,
                angle.positive,
                angular_velocity.positive,
            ): action.nothing,
            # ========== POSITION: ZERO ==========
            # Position: zero, Velocity: negative
            (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.negative,
            ): action.strong_left,
            (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ): action.left,
            (
                position.zero,
                velocity.negative,
                angle.negative,
                angular_velocity.positive,
            ): action.left,
            (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ): action.left,
            (
                position.zero,
                velocity.negative,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.zero,
                velocity.negative,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.zero,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ): action.left,
            (
                position.zero,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: zero, Velocity: zero
            (
                position.zero,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ): action.left,
            (
                position.zero,
                velocity.zero,
                angle.negative,
                angular_velocity.zero,
            ): action.left,
            (
                position.zero,
                velocity.zero,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.zero,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.zero,
                velocity.zero,
                angle.zero,
                angular_velocity.zero,
            ): action.nothing,
            (
                position.zero,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.zero,
                velocity.zero,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.zero,
                velocity.zero,
                angle.positive,
                angular_velocity.zero,
            ): action.right,
            (
                position.zero,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: zero, Velocity: positive
            (
                position.zero,
                velocity.positive,
                angle.negative,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.zero,
                velocity.positive,
                angle.negative,
                angular_velocity.zero,
            ): action.nothing,
            (
                position.zero,
                velocity.positive,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.zero,
                velocity.positive,
                angle.zero,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.zero,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ): action.right,
            (
                position.zero,
                velocity.positive,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.zero,
                velocity.positive,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.zero,
                velocity.positive,
                angle.positive,
                angular_velocity.zero,
            ): action.right,
            (
                position.zero,
                velocity.positive,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # ========== POSITION: POSITIVE ==========
            # Position: positive, Velocity: negative
            (
                position.positive,
                velocity.negative,
                angle.negative,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.positive,
                velocity.negative,
                angle.negative,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.negative,
                angle.negative,
                angular_velocity.positive,
            ): action.right,
            (
                position.positive,
                velocity.negative,
                angle.zero,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.positive,
                velocity.negative,
                angle.zero,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.negative,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.positive,
                velocity.negative,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.positive,
                velocity.negative,
                angle.positive,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.negative,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: positive, Velocity: zero
            (
                position.positive,
                velocity.zero,
                angle.negative,
                angular_velocity.negative,
            ): action.left,
            (
                position.positive,
                velocity.zero,
                angle.negative,
                angular_velocity.zero,
            ): action.nothing,
            (
                position.positive,
                velocity.zero,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.negative,
            ): action.right,
            (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.zero,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.positive,
                velocity.zero,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.positive,
                velocity.zero,
                angle.positive,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.zero,
                angle.positive,
                angular_velocity.positive,
            ): action.right,
            # Position: positive, Velocity: positive
            (
                position.positive,
                velocity.positive,
                angle.negative,
                angular_velocity.negative,
            ): action.left,
            (
                position.positive,
                velocity.positive,
                angle.negative,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.positive,
                angle.negative,
                angular_velocity.positive,
            ): action.nothing,
            (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.negative,
            ): action.left,
            (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.positive,
                angle.zero,
                angular_velocity.positive,
            ): action.right,
            (
                position.positive,
                velocity.positive,
                angle.positive,
                angular_velocity.negative,
            ): action.nothing,
            (
                position.positive,
                velocity.positive,
                angle.positive,
                angular_velocity.zero,
            ): action.right,
            (
                position.positive,
                velocity.positive,
                angle.positive,
                angular_velocity.positive,
            ): action.strong_right,
        }
    )

    return rules
