"""Active matter: self-propelled particles, active clusters, flocking, and polar active fluids."""

from ._alignment import PolarAlignment, aligning_active_brownian_particles
from ._collective import (
    CuckerSmaleAlignment,
    PhoreticAttraction,
    PolarFluid,
    SoftRepulsion,
    cucker_smale_flocking,
    living_crystals,
    motility_induced_phase_separation,
    toner_tu,
)
from ._collisions import BinaryCollisions, Flocker, binary_collision_flocking
from ._particles import (
    ActiveParticle,
    DepotParticle,
    OrnsteinUhlenbeckParticle,
    Tumbler,
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    run_and_tumble,
)

__all__: list[str] = [
    'ActiveParticle',
    'BinaryCollisions',
    'CuckerSmaleAlignment',
    'DepotParticle',
    'Flocker',
    'OrnsteinUhlenbeckParticle',
    'PhoreticAttraction',
    'PolarAlignment',
    'PolarFluid',
    'SoftRepulsion',
    'Tumbler',
    'active_brownian_particles',
    'active_ornstein_uhlenbeck_particles',
    'aligning_active_brownian_particles',
    'binary_collision_flocking',
    'chiral_active_particles',
    'cucker_smale_flocking',
    'depot_brownian_particles',
    'living_crystals',
    'motility_induced_phase_separation',
    'run_and_tumble',
    'toner_tu',
]
