"""Soft matter: Brownian particles, optical traps, anomalous diffusion, and noise-driven transitions."""

from ._brownian import DoubleWell, WallParticle, brownian_near_wall, stochastic_resonance
from ._diffusion import (
    BrownianParticle,
    ModulatedParticle,
    OpticalTrap,
    ScaledWalker,
    SpeckleField,
    brownian_motion,
    diffusion_in_a_speckle_field,
    kramers_escape,
    multiplicative_noise,
    optical_trap,
    scaled_brownian_motion,
)

__all__: list[str] = [
    'BrownianParticle',
    'DoubleWell',
    'ModulatedParticle',
    'OpticalTrap',
    'ScaledWalker',
    'SpeckleField',
    'WallParticle',
    'brownian_motion',
    'brownian_near_wall',
    'diffusion_in_a_speckle_field',
    'kramers_escape',
    'multiplicative_noise',
    'optical_trap',
    'scaled_brownian_motion',
    'stochastic_resonance',
]
