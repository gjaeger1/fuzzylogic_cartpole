"""
Fuzzy Set Visualization Module

This module provides functionality to visualize fuzzy sets from YAML specification files.
It creates plots showing all fuzzy sets for each domain, making it easy to understand
the fuzzy logic configuration.

Usage:
    # As a module
    from fuzzylogic_cartpole.visualize_fuzzy_sets import visualize_fuzzy_sets
    visualize_fuzzy_sets('path/to/fuzzy_sets.yaml')

    # From command line
    python -m fuzzylogic_cartpole.visualize_fuzzy_sets path/to/fuzzy_sets.yaml
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from .rule_base_generation import (
    generate_domains,
    generate_fuzzy_sets,
    load_specification,
)


def visualize_fuzzy_sets(yaml_file, save_path=None, show=True, figsize=None):
    """
    Visualize fuzzy sets from a YAML specification file.

    Creates one subplot for each domain, showing all fuzzy sets defined on that domain.

    Args:
        yaml_file (str or Path): Path to the YAML file containing fuzzy set specifications
        save_path (str or Path, optional): If provided, save the figure to this path
        show (bool): Whether to display the plot (default: True)
        figsize (tuple, optional): Figure size as (width, height). If None, auto-calculated

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Load specifications from YAML
    domain_specs, fuzzy_set_specs, rule_specs, default_outputs = load_specification(
        yaml_file
    )

    # Generate domains and fuzzy sets
    domains = generate_domains(domain_specs)
    domains = generate_fuzzy_sets(domains, fuzzy_set_specs)

    # Organize fuzzy sets by domain
    domain_fuzzy_sets = {}
    for domain in domains:
        domain_name = domain._name
        # Get all fuzzy sets for this domain
        fuzzy_set_names = []
        for spec in fuzzy_set_specs:
            if spec["domain"] == domain_name:
                fuzzy_set_names.append(spec["name"])
        domain_fuzzy_sets[domain_name] = fuzzy_set_names

    # Calculate figure layout
    n_domains = len(domains)
    n_cols = min(2, n_domains)  # Max 2 columns
    n_rows = (n_domains + n_cols - 1) // n_cols  # Ceiling division

    # Set figure size
    if figsize is None:
        fig_width = 7 * n_cols
        fig_height = 4 * n_rows
        figsize = (fig_width, fig_height)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f"Fuzzy Sets Visualization: {Path(yaml_file).name}",
        fontsize=14,
        fontweight="bold",
    )

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Plot each domain
    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        domain_name = domain._name

        # Get domain range
        x_min = domain._low
        x_max = domain._high
        resolution = domain._res

        # Create x values for plotting
        x_values = np.arange(x_min, x_max + resolution, resolution)

        # Plot each fuzzy set for this domain
        fuzzy_set_names = domain_fuzzy_sets.get(domain_name, [])

        if fuzzy_set_names:
            for set_name in fuzzy_set_names:
                # Get the fuzzy set object
                fuzzy_set = getattr(domain, set_name)

                # Evaluate membership function for all x values
                y_values = []
                for x in x_values:
                    try:
                        membership = fuzzy_set(x)
                        y_values.append(membership)
                    except Exception:
                        y_values.append(0.0)

                # Plot the fuzzy set
                ax.plot(x_values, y_values, label=set_name, linewidth=2)

            # Set labels and title
            ax.set_xlabel(domain_name.replace("_", " ").title(), fontsize=11)
            ax.set_ylabel("Membership Degree", fontsize=11)
            ax.set_title(
                f"{domain_name.replace('_', ' ').title()} Domain",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
        else:
            # No fuzzy sets for this domain
            ax.text(
                0.5,
                0.5,
                f"No fuzzy sets\nfor {domain_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )
            ax.set_xlabel(domain_name.replace("_", " ").title())
            ax.set_title(
                f"{domain_name.replace('_', ' ').title()} Domain",
                fontsize=12,
                fontweight="bold",
            )

    # Hide any unused subplots
    for idx in range(n_domains, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


@click.command()
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option(
    "--save",
    "-s",
    "save_path",
    type=click.Path(),
    help="Path to save the figure (e.g., fuzzy_sets.png)",
)
@click.option(
    "--no-show", is_flag=True, help="Do not display the plot (useful when only saving)"
)
@click.option(
    "--width",
    type=float,
    default=None,
    help="Figure width in inches (default: auto-calculated)",
)
@click.option(
    "--height",
    type=float,
    default=None,
    help="Figure height in inches (default: auto-calculated)",
)
def main(yaml_file, save_path, no_show, width, height):
    """
    Visualize fuzzy sets from a YAML specification file.

    YAML_FILE: Path to the YAML file containing domain and fuzzy set specifications.

    Examples:

        # Display the visualization
        python -m fuzzylogic_cartpole.visualize_fuzzy_sets optimized_fuzzy_sets.yaml

        # Save to file without displaying
        python -m fuzzylogic_cartpole.visualize_fuzzy_sets optimized_fuzzy_sets.yaml -s fuzzy_sets.png --no-show

        # Custom figure size
        python -m fuzzylogic_cartpole.visualize_fuzzy_sets optimized_fuzzy_sets.yaml --width 14 --height 10
    """
    figsize = None
    if width is not None and height is not None:
        figsize = (width, height)
    elif width is not None or height is not None:
        click.echo(
            "Warning: Both --width and --height must be specified. Ignoring custom size.",
            err=True,
        )

    show = not no_show

    try:
        visualize_fuzzy_sets(yaml_file, save_path=save_path, show=show, figsize=figsize)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
