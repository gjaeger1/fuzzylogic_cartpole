"""
Fuzzy Logic CartPole Controller

A simple example on how to use fuzzy logic to control the CartPole gymnasium environment.
"""

__version__ = "0.1.0"

from .controller import (
    FuzzyCartPoleController,
    get_standard_domains,
    get_standard_rules,
)

__all__ = ["FuzzyCartPoleController", "get_standard_domains", "get_standard_rules"]
