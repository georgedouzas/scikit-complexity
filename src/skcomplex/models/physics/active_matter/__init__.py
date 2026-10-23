"""Active matter: self-propelled particles, clusters, flocking, polar fluids, chemotaxis, and Active Model B+."""

from ._alignment import PolarAlignment, aligning_active_brownian_particles
from ._chemotaxis import Chemotaxis, keller_segel
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
from ._flocks import CohesiveParticle, NematicParticle, VicsekParticle, ginelli_nematics, gregoire_chate, vicsek
from ._model_b import ActiveScalar, active_model_b
from ._nonreciprocal import NonreciprocalMover, nonreciprocal_vicsek
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
    'ActiveScalar',
    'BinaryCollisions',
    'Chemotaxis',
    'CohesiveParticle',
    'CuckerSmaleAlignment',
    'DepotParticle',
    'Flocker',
    'NematicParticle',
    'NonreciprocalMover',
    'OrnsteinUhlenbeckParticle',
    'PhoreticAttraction',
    'PolarAlignment',
    'PolarFluid',
    'SoftRepulsion',
    'Tumbler',
    'VicsekParticle',
    'active_brownian_particles',
    'active_model_b',
    'active_ornstein_uhlenbeck_particles',
    'aligning_active_brownian_particles',
    'binary_collision_flocking',
    'chiral_active_particles',
    'cucker_smale_flocking',
    'depot_brownian_particles',
    'ginelli_nematics',
    'gregoire_chate',
    'keller_segel',
    'living_crystals',
    'motility_induced_phase_separation',
    'nonreciprocal_vicsek',
    'run_and_tumble',
    'toner_tu',
    'vicsek',
]
