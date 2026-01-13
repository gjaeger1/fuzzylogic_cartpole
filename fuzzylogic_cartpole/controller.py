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

    def __init__(self):
        """Initialize the fuzzy logic controller with membership functions and rules."""
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        """Set up the fuzzy logic system with domains, sets, and rules."""

        # Define input domains
        # Cart position: typically -2.4 to 2.4
        self.position = Domain("position", -4.8, 4.8, res=0.01)
        self.position.negative = S(-1.0, 0)  # S(-
        self.position.zero = triangular(-0.5, 0.5, c=0.0)
        self.position.positive = R(0, 1.0)  # R(0, 4.0)

        # Cart velocity: typically -inf to inf, but practically -2 to 2
        self.velocity = Domain("velocity", -4.0, 4.0, res=0.01)
        self.velocity.negative = S(-4.0, -0.5)
        self.velocity.zero = triangular(-1.0, 1.0, c=0.0)
        self.velocity.positive = R(0.5, 4.0)

        # Pole angle: typically -0.418 to 0.418 radians (~24 degrees)
        self.angle = Domain("angle", -0.5, 0.5, res=0.01)
        self.angle.negative = S(-0.5, -0.05)
        self.angle.zero = triangular(-0.1, 0.1, c=0.0)
        self.angle.positive = R(0.05, 0.5)
        # Pole angular velocity: typically -inf to inf, but practically -2 to 2
        self.angular_velocity = Domain("angular_velocity", -4.0, 4.0, res=0.01)
        self.angular_velocity.negative = S(-4.0, -0.5)
        self.angular_velocity.zero = triangular(-0.5, 0.5, c=0.0)
        self.angular_velocity.positive = R(0.5, 4.0)

        # Define output domain
        # Action: 0 (push left) or 1 (push right)
        # We'll use a continuous output and threshold at 0.5
        self.action = Domain("action", 0.0, 1.0, res=0.01)
        self.action.strong_left = S(0.0, 0.25)
        self.action.left = triangular(0.1, 0.5, c=0.25)
        self.action.nothing = triangular(0.425, 0.575, c=0.5)
        self.action.right = triangular(0.5, 0.9, c=0.75)
        self.action.strong_right = R(0.75, 1.0)

        # Define fuzzy rules
        # The main goal is to keep the pole upright (angle close to zero)
        # and the cart near the center (position close to zero)
        # Combine all rules into a single Rule object
        self.rules = Rule(
            {
                # Complete rule base covering all 81 combinations (3^4)
                # Format: (position, velocity, angle, angular_velocity): action
                # Priority: Angle control > Angular velocity > Position > Velocity
                # ========== POSITION: NEGATIVE ==========
                # Position: negative, Velocity: negative
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.strong_left,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: negative, Velocity: zero
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: negative, Velocity: positive
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.strong_left,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.nothing,
                (
                    self.position.negative,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                # ========== POSITION: ZERO ==========
                # Position: zero, Velocity: negative
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.strong_left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: zero, Velocity: zero
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: zero, Velocity: positive
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.zero,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # ========== POSITION: POSITIVE ==========
                # Position: positive, Velocity: negative
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.negative,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: positive, Velocity: zero
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.zero,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.right,
                # Position: positive, Velocity: positive
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.negative,
                    self.angular_velocity.positive,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.negative,
                ): self.action.left,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.zero,
                    self.angular_velocity.positive,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.negative,
                ): self.action.nothing,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.zero,
                ): self.action.right,
                (
                    self.position.positive,
                    self.velocity.positive,
                    self.angle.positive,
                    self.angular_velocity.positive,
                ): self.action.strong_right,
            }
        )

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

        # Create values dictionary mapping domains to observation values
        values = {
            self.position: cart_position,
            self.velocity: cart_velocity,
            self.angle: pole_angle,
            self.angular_velocity: pole_angular_velocity,
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
