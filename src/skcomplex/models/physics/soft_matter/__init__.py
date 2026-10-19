"""Soft matter: Brownian particles and noise-driven transitions."""

from ._brownian import DoubleWell, WallParticle, brownian_near_wall, stochastic_resonance

__all__: list[str] = [
    'DoubleWell',
    'WallParticle',
    'brownian_near_wall',
    'stochastic_resonance',
]
