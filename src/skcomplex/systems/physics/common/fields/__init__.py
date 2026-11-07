"""Field realizations shared by the physics packages: Monte Carlo sampling of lattice gauge fields."""

from ._wilson_sampling import WilsonSamplingInteraction

__all__: list[str] = [
    'WilsonSamplingInteraction',
]
