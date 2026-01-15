"""
Fuzzy Logic Controller for CartPole Environment
"""

import numpy as np
import yaml
from fuzzylogic.classes import Domain, Rule, Set
from fuzzylogic.functions import R, S, trapezoid, triangular

from .controller import FuzzyCartPoleController
from .defaults import (
    get_standard_domain_specs,
    get_standard_fuzzy_sets_specs,
    get_standard_rules_spec,
    get_standard_specifications,
)


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


def generate_rule_base(domains, specifications, default_outputs=None):
    """Generate fuzzy rules

    Args:
        domains: List of Domain objects (both inputs and outputs)
        specifications: List of rule specifications with "inputs" as strings in format
                       ["domain_name.fuzzyset_name", ...] or a tuple/list of such strings
        default_outputs: Dict mapping output domain names to default fuzzy set names
    """
    # For each combination of possible fuzzy sets on the input_domains there need to be a rule specified in the specifications. Otherwise, if given, default outputs need to be assigned.
    # if multiple output domains are present, separate rule bases are generated. One for each output domain

    import itertools

    # Convert to list if not already
    if not isinstance(domains, (list, tuple)):
        domains = [domains]

    # Create a dictionary to easily access domains by name
    domain_dict = {domain._name: domain for domain in domains}

    # Parse specifications to identify output domains
    output_domain_names = set()
    for spec in specifications:
        output_domain_name = spec.get("output_domain")
        if output_domain_name:
            output_domain_names.add(output_domain_name)

    # Separate input and output domains
    output_domains = [d for d in domains if d._name in output_domain_names]
    input_domains = [d for d in domains if d._name not in output_domain_names]

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
                    # spec_inputs should be a list/tuple of strings like ["position.negative", "velocity.zero", ...]
                    spec_inputs = spec.get("inputs")

                    # Convert string specifications to fuzzy set objects
                    if isinstance(spec_inputs, (list, tuple)):
                        try:
                            spec_fuzzy_sets = []
                            for input_spec in spec_inputs:
                                # Parse "domain_name.fuzzyset_name"
                                if isinstance(input_spec, str) and "." in input_spec:
                                    domain_name, set_name = input_spec.split(".", 1)
                                    domain = domain_dict[domain_name]
                                    fuzzy_set = getattr(domain, set_name)
                                    spec_fuzzy_sets.append(fuzzy_set)
                                else:
                                    # Backwards compatibility: if it's already a Set object
                                    spec_fuzzy_sets.append(input_spec)

                            # Check if this matches the current combination
                            if tuple(spec_fuzzy_sets) == combo:
                                # Found a matching rule
                                output_set_name = spec.get("output")
                                output_set = getattr(output_domain, output_set_name)
                                rules_dict[combo] = output_set
                                rule_found = True
                                break
                        except (KeyError, AttributeError, ValueError):
                            # Skip this spec if it has invalid domain/set names
                            continue

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
    specification = get_standard_domain_specs()

    return generate_domains(specification)


def get_standard_fuzzy_sets(position, velocity, angle, angular_velocity, action):
    # Define specifications for all fuzzy sets
    specifications = get_standard_fuzzy_sets_specs()

    # Use generate_fuzzy_sets to create all fuzzy sets
    domains = (position, velocity, angle, angular_velocity, action)
    generate_fuzzy_sets(domains, specifications)

    return position, velocity, angle, angular_velocity, action


def get_standard_rules(position, velocity, angle, angular_velocity, action):
    # Define specifications for rules that differ from the default (action.nothing)
    specifications = get_standard_rules_spec()

    # Use generate_rule_base with action.nothing as default
    domains = [position, velocity, angle, angular_velocity, action]
    default_outputs = {"action": "nothing"}

    rules = generate_rule_base(domains, specifications, default_outputs)

    return rules


def save_specification(
    domain_specs, fuzzy_set_specs, rule_specs, filename, default_outputs=None
):
    """Write specifications to YAML file

    Args:
        domain_specs: List of domain specifications (for generate_domains)
        fuzzy_set_specs: List of fuzzy set specifications (for generate_fuzzy_sets)
        rule_specs: List of rule specifications (for generate_rule_base)
        filename: Path to the YAML file to write
        default_outputs: Optional dict mapping output domain names to default fuzzy set names
    """
    specification_data = {
        "domains": domain_specs,
        "fuzzy_sets": fuzzy_set_specs,
        "rules": rule_specs,
    }

    if default_outputs is not None:
        specification_data["default_outputs"] = default_outputs

    with open(filename, "w") as f:
        yaml.dump(specification_data, f, default_flow_style=False, sort_keys=False)


def load_specification(filename):
    """Load specifications from YAML file

    Args:
        filename: Path to the YAML file to read

    Returns:
        tuple: (domain_specs, fuzzy_set_specs, rule_specs, default_outputs)
               default_outputs will be None if not present in the YAML file
    """
    with open(filename, "r") as f:
        specification_data = yaml.safe_load(f)

    domain_specs = specification_data.get("domains", [])
    fuzzy_set_specs = specification_data.get("fuzzy_sets", [])
    rule_specs = specification_data.get("rules", [])
    default_outputs = specification_data.get("default_outputs", None)

    return domain_specs, fuzzy_set_specs, rule_specs, default_outputs


def generate_controller_from_file(filename):
    domain_specs, fuzzy_set_specs, rule_specs, default_outputs = load_specification(
        filename
    )

    # Generate domains, fuzzy sets, and rules
    domains = generate_fuzzy_sets(generate_domains(domain_specs), fuzzy_set_specs)
    rules = generate_rule_base(
        domains,
        rule_specs,
        default_outputs,
    )

    # Create fuzzy controller with loaded configuration
    return FuzzyCartPoleController(domains, rules)
