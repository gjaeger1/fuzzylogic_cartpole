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
    # Define fuzzy sets for position
    position.negative = S(-1.0, 0)  # S(-
    position.zero = triangular(-0.5, 0.5, c=0.0)
    position.positive = R(0, 1.0)  # R(0, 4.0)

    # Define fuzzy sets for velocity
    velocity.negative = S(-4.0, -0.5)
    velocity.zero = triangular(-1.0, 1.0, c=0.0)
    velocity.positive = R(0.5, 4.0)

    # Define fuzzy sets for angle
    angle.negative = S(-0.5, -0.05)
    angle.zero = triangular(-0.1, 0.1, c=0.0)
    angle.positive = R(0.05, 0.5)

    # Define fuzzy sets for angular velocity
    angular_velocity.negative = S(-4.0, -0.5)
    angular_velocity.zero = triangular(-0.5, 0.5, c=0.0)
    angular_velocity.positive = R(0.5, 4.0)

    # Define fuzzy sets for action
    action.strong_left = S(0.0, 0.25)
    action.left = triangular(0.1, 0.5, c=0.25)
    action.nothing = triangular(0.425, 0.575, c=0.5)
    action.right = triangular(0.5, 0.9, c=0.75)
    action.strong_right = R(0.75, 1.0)

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
