"""Active matter: self-propelled particles, clusters, flocking, polar fluids, chemotaxis, and Active Model B+."""

from ._alignment import PolarAlignmentInteraction, aligning_active_brownian_particles
from ._chemotaxis import ChemotaxisComponent, keller_segel
from ._collective import (
    CuckerSmaleAlignmentInteraction,
    PhoreticAttractionInteraction,
    PolarFluidComponent,
    SoftRepulsionInteraction,
    cucker_smale_flocking,
    living_crystals,
    motility_induced_phase_separation,
    toner_tu,
)
from ._collisions import BinaryCollisionsInteraction, FlockerComponent, binary_collision_flocking
from ._flocks import (
    CohesiveParticleComponent,
    NematicParticleComponent,
    VicsekParticleComponent,
    ginelli_nematics,
    gregoire_chate,
    vicsek,
)
from ._model_b import ActiveScalarComponent, active_model_b
from ._nonreciprocal import NonreciprocalMoverComponent, nonreciprocal_vicsek
from ._particles import (
    ActiveParticleComponent,
    DepotParticleComponent,
    OrnsteinUhlenbeckParticleComponent,
    TumblerComponent,
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    run_and_tumble,
)

__all__: list[str] = [
    'ActiveParticleComponent',
    'ActiveScalarComponent',
    'BinaryCollisionsInteraction',
    'ChemotaxisComponent',
    'CohesiveParticleComponent',
    'CuckerSmaleAlignmentInteraction',
    'DepotParticleComponent',
    'FlockerComponent',
    'NematicParticleComponent',
    'NonreciprocalMoverComponent',
    'OrnsteinUhlenbeckParticleComponent',
    'PhoreticAttractionInteraction',
    'PolarAlignmentInteraction',
    'PolarFluidComponent',
    'SoftRepulsionInteraction',
    'TumblerComponent',
    'VicsekParticleComponent',
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
