"""Soft matter: Brownian particles, optical traps, anomalous diffusion, and noise-driven transitions."""

from ._brownian import DoubleWell, WallParticle, brownian_near_wall, stochastic_resonance
from ._colored_noise import ColoredNoiseParticle, colored_noise_langevin
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
from ._friction import FrictionalBlock, shaken_spring_block
from ._ratchets import FlashingSawtooth, TiltedWashboard, flashing_ratchet, tilted_periodic_potential
from ._traps import TrapSite, biased_trap_walk

__all__: list[str] = [
    'BrownianParticle',
    'ColoredNoiseParticle',
    'DoubleWell',
    'FlashingSawtooth',
    'FrictionalBlock',
    'ModulatedParticle',
    'OpticalTrap',
    'ScaledWalker',
    'SpeckleField',
    'TiltedWashboard',
    'TrapSite',
    'WallParticle',
    'biased_trap_walk',
    'brownian_motion',
    'brownian_near_wall',
    'colored_noise_langevin',
    'diffusion_in_a_speckle_field',
    'flashing_ratchet',
    'kramers_escape',
    'multiplicative_noise',
    'optical_trap',
    'scaled_brownian_motion',
    'shaken_spring_block',
    'stochastic_resonance',
    'tilted_periodic_potential',
]
