"""
Fuzzy Logic Controller for CartPole Environment
"""

import numpy as np
from fuzzylogic.classes import Domain, Rule, Set
from fuzzylogic.functions import R, S, trapezoid, triangular


def get_standard_domains():
    # Define input domains
    # Cart position: typically -2.4 to 2.4
    position = Domain("position", -4.8, 4.8, res=0.01)
    position.negative = S(-1.0, 0)  # S(-
    position.zero = triangular(-0.5, 0.5, c=0.0)
    position.positive = R(0, 1.0)  # R(0, 4.0)

    # Cart velocity: typically -inf to inf, but practically -2 to 2
    velocity = Domain("velocity", -4.0, 4.0, res=0.01)
    velocity.negative = S(-4.0, -0.5)
    velocity.zero = triangular(-1.0, 1.0, c=0.0)
    velocity.positive = R(0.5, 4.0)

    # Pole angle: typically -0.418 to 0.418 radians (~24 degrees)
    angle = Domain("angle", -0.5, 0.5, res=0.01)
    angle.negative = S(-0.5, -0.05)
    angle.zero = triangular(-0.1, 0.1, c=0.0)
    angle.positive = R(0.05, 0.5)
    # Pole angular velocity: typically -inf to inf, but practically -2 to 2
    angular_velocity = Domain("angular_velocity", -4.0, 4.0, res=0.01)
    angular_velocity.negative = S(-4.0, -0.5)
    angular_velocity.zero = triangular(-0.5, 0.5, c=0.0)
    angular_velocity.positive = R(0.5, 4.0)

    # Define output domain
    # Action: 0 (push left) or 1 (push right)
    # We'll use a continuous output and threshold at 0.5
    action = Domain("action", 0.0, 1.0, res=0.01)
    action.strong_left = S(0.0, 0.25)
    action.left = triangular(0.1, 0.5, c=0.25)
    action.nothing = triangular(0.425, 0.575, c=0.5)
    action.right = triangular(0.5, 0.9, c=0.75)
    action.strong_right = R(0.75, 1.0)

    return [position, velocity, angle, angular_velocity, action]


def get_standard_rules():
    position, velocity, angle, angular_velocity, action = get_standard_domains()
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


class FuzzyCartPoleController:
    """
    A fuzzy logic controller for the CartPole-v1 environment.

    The controller uses fuzzy logic to determine the action based on
    the cart position, cart velocity, pole angle, and pole angular velocity.
    """

    def __init__(self, domains=get_standard_domains(), rules=get_standard_rules()):
        """Initialize the fuzzy logic controller with membership functions and rules."""
        self.domains = domains
        self.rules = rules

    def get_action(self, observation):
        """
        Determine the action based on the current observation.

        Args:
            observation: A tuple/array of (cart_position, cart_velocity, pole_angle, pole_angular_velocity)

        Returns:
            int: Action to take (0 for left, 1 for right)
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation

        print("Observation:")
        print(observation)
        position = Domain("position", -4.8, 4.8, res=0.01)
        position.negative = S(-1.0, 0)  # S(-
        position.zero = triangular(-0.5, 0.5, c=0.0)
        position.positive = R(0, 1.0)  # R(0, 4.0)

        # Create values dictionary mapping domains to observation values
        values = {
            self.domains[0]: cart_position,
            self.domains[1]: cart_velocity,
            self.domains[2]: pole_angle,
            self.domains[3]: pole_angular_velocity,
        }

        # Use the library's built-in inference and defuzzification
        fuzzy_output = self.rules(values)

        print("Fuzzy output:")
        print(fuzzy_output)

        # Handle case where no rules fired
        if fuzzy_output is None:
            fuzzy_output = 0.5

        # Convert to discrete action
        action = 1 if fuzzy_output > 0.5 else 0

        return action
