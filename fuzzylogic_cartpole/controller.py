"""
Fuzzy Logic Controller for CartPole Environment
"""

import numpy as np
from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.functions import R, S, triangular


class FuzzyCartPoleController:
    """
    A fuzzy logic controller for the CartPole-v1 environment.
    
    The controller uses fuzzy logic to determine the action based on
    the cart position, cart velocity, pole angle, and pole angular velocity.
    """
    
    # Output values for defuzzification
    LEFT_ACTION_VALUE = 0.15
    RIGHT_ACTION_VALUE = 0.85
    NEUTRAL_ACTION_VALUE = 0.5
    
    def __init__(self):
        """Initialize the fuzzy logic controller with membership functions and rules."""
        self._setup_fuzzy_system()
    
    def _setup_fuzzy_system(self):
        """Set up the fuzzy logic system with domains, sets, and rules."""
        
        # Define input domains
        # Cart position: typically -2.4 to 2.4
        self.position = Domain("position", -3.0, 3.0)
        self.position.negative = S(-3.0, -0.5)
        self.position.zero = triangular(-1.0, 1.0, c=0.0)
        self.position.positive = R(0.5, 3.0)
        
        # Cart velocity: typically -inf to inf, but practically -2 to 2
        self.velocity = Domain("velocity", -4.0, 4.0)
        self.velocity.negative = S(-4.0, -0.5)
        self.velocity.zero = triangular(-1.0, 1.0, c=0.0)
        self.velocity.positive = R(0.5, 4.0)
        
        # Pole angle: typically -0.418 to 0.418 radians (~24 degrees)
        self.angle = Domain("angle", -0.5, 0.5)
        self.angle.negative = S(-0.5, -0.05)
        self.angle.zero = triangular(-0.1, 0.1, c=0.0)
        self.angle.positive = R(0.05, 0.5)
        
        # Pole angular velocity: typically -inf to inf, but practically -2 to 2
        self.angular_velocity = Domain("angular_velocity", -4.0, 4.0)
        self.angular_velocity.negative = S(-4.0, -0.5)
        self.angular_velocity.zero = triangular(-1.0, 1.0, c=0.0)
        self.angular_velocity.positive = R(0.5, 4.0)
        
        # Define output domain
        # Action: 0 (push left) or 1 (push right)
        # We'll use a continuous output and threshold at 0.5
        self.action = Domain("action", 0.0, 1.0)
        self.action.left = S(0.0, 0.3)
        self.action.neutral = triangular(0.3, 0.7, c=0.5)
        self.action.right = R(0.7, 1.0)
        
        # Define fuzzy rules
        # The main goal is to keep the pole upright (angle close to zero)
        # and the cart near the center (position close to zero)
        self.rules = [
            # If pole is falling left (negative angle), push left to counteract
            Rule({(self.angle.negative,): self.action.left}),
            
            # If pole is falling right (positive angle), push right to counteract
            Rule({(self.angle.positive,): self.action.right}),
            
            # If pole is upright but moving left, push right to stabilize
            Rule({(self.angle.zero, self.angular_velocity.negative): self.action.right}),
            
            # If pole is upright but moving right, push left to stabilize
            Rule({(self.angle.zero, self.angular_velocity.positive): self.action.left}),
            
            # If pole is upright and stable, balance based on cart position
            Rule({(self.angle.zero, self.angular_velocity.zero, self.position.negative): self.action.right}),
            Rule({(self.angle.zero, self.angular_velocity.zero, self.position.positive): self.action.left}),
            
            # Additional rules for cart velocity
            Rule({(self.angle.zero, self.position.zero, self.velocity.negative): self.action.left}),
            Rule({(self.angle.zero, self.position.zero, self.velocity.positive): self.action.right}),
        ]
    
    def get_action(self, observation):
        """
        Determine the action based on the current observation.
        
        Args:
            observation: A tuple/array of (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
        
        Returns:
            int: Action to take (0 for left, 1 for right)
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        
        # Evaluate fuzzy rules
        fuzzy_output = 0.0  # Initialize accumulator for weighted output
        total_weight = 0.0
        
        for rule in self.rules:
            # Get the conditions from the rule
            for antecedent, consequent in rule.conditions.items():
                # Calculate membership degree for this rule
                membership = 1.0
                
                for input_set in antecedent:
                    if input_set.domain == self.position:
                        membership = min(membership, input_set(cart_position))
                    elif input_set.domain == self.velocity:
                        membership = min(membership, input_set(cart_velocity))
                    elif input_set.domain == self.angle:
                        membership = min(membership, input_set(pole_angle))
                    elif input_set.domain == self.angular_velocity:
                        membership = min(membership, input_set(pole_angular_velocity))
                
                # Accumulate weighted output
                if membership > 0:
                    # Use center of gravity for the consequent set
                    if consequent == self.action.left:
                        output_value = self.LEFT_ACTION_VALUE
                    elif consequent == self.action.right:
                        output_value = self.RIGHT_ACTION_VALUE
                    else:
                        output_value = self.NEUTRAL_ACTION_VALUE
                    
                    fuzzy_output += membership * output_value
                    total_weight += membership
        
        # Defuzzification: weighted average
        if total_weight > 0:
            fuzzy_output /= total_weight
        else:
            # Default to neutral if no rules fired
            fuzzy_output = self.NEUTRAL_ACTION_VALUE
        
        # Convert to discrete action
        action = 1 if fuzzy_output > 0.5 else 0
        
        return action
