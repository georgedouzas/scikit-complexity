"""Field realizations shared by the physics packages.

The package holds the Monte Carlo sampling of lattice gauge fields.
"""

from ._wilson_sampling import WilsonSamplingInteraction

__all__: list[str] = [
    'WilsonSamplingInteraction',
]
