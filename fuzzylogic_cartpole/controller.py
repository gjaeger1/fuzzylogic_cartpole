"""
Fuzzy Logic Controller for CartPole Environment
"""

import numpy as np
from fuzzylogic.classes import Domain, Rule, Set
from fuzzylogic.functions import R, S, trapezoid, triangular


class FuzzyCartPoleController:
    """
    A fuzzy logic controller for the CartPole-v1 environment.

    The controller uses fuzzy logic to determine the action based on
    the cart position, cart velocity, pole angle, and pole angular velocity.
    """

    def __init__(self, domains=None, rules=None, verbose=False):
        """Initialize the fuzzy logic controller with membership functions and rules.

        Args:
            domains: Tuple of fuzzy logic domains
            rules: Fuzzy logic rules
            verbose: If True, print debug information during action selection
        """
        self.domains = domains
        self.rules = rules
        self.verbose = verbose

    def get_action(self, observation):
        """
        Determine the action based on the current observation.

        Args:
            observation: A tuple/array of (cart_position, cart_velocity, pole_angle, pole_angular_velocity)

        Returns:
            int: Action to take (0 for left, 1 for right)
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation

        if self.verbose:
            print("Observation:")
            print(observation)

        # Create values dictionary mapping domains to observation values
        values = {
            self.domains[0]: cart_position,
            self.domains[1]: cart_velocity,
            self.domains[2]: pole_angle,
            self.domains[3]: pole_angular_velocity,
        }

        # Use the library's built-in inference and defuzzification
        fuzzy_output = self.rules(values)

        if self.verbose:
            print("Fuzzy output:")
            print(fuzzy_output)

        # Handle case where no rules fired
        if fuzzy_output is None:
            fuzzy_output = 0.5

        # Convert to discrete action
        action = 1 if fuzzy_output > 0.5 else 0

        return action
