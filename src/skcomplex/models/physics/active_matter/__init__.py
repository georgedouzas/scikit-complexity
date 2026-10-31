"""Active matter: self-propelled particles, flocking, polar fluids, chemotaxis, Active Model B+, and their orders."""

from ._alignment import PolarAlignmentInteraction, aligning_active_brownian_particles
from ._chemotaxis import ChemotaxisElement, keller_segel
from ._collective import (
    CuckerSmaleAlignmentInteraction,
    PhoreticAttractionInteraction,
    PolarFluidElement,
    cucker_smale_flocking,
    living_crystals,
    motility_induced_phase_separation,
    toner_tu,
)
from ._collisions import BinaryCollisionsInteraction, FlockerElement, binary_collision_flocking
from ._flocker_view import FlockerView
from ._flocks import (
    CohesiveParticleElement,
    NematicParticleElement,
    VicsekParticleElement,
    ginelli_nematics,
    gregoire_chate,
    vicsek,
)
from ._heading_alignment import HeadingAlignmentInteraction
from ._model_b import ActiveScalarElement, active_model_b
from ._nonreciprocal import NonreciprocalMoverElement, nonreciprocal_vicsek
from ._particles import (
    ActiveParticleElement,
    DepotParticleElement,
    OrnsteinUhlenbeckParticleElement,
    TumblerElement,
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    run_and_tumble,
)
from ._tumbler_view import TumblerView

__all__: list[str] = [
    'ActiveParticleElement',
    'ActiveScalarElement',
    'BinaryCollisionsInteraction',
    'ChemotaxisElement',
    'CohesiveParticleElement',
    'CuckerSmaleAlignmentInteraction',
    'DepotParticleElement',
    'FlockerElement',
    'FlockerView',
    'HeadingAlignmentInteraction',
    'NematicParticleElement',
    'NonreciprocalMoverElement',
    'OrnsteinUhlenbeckParticleElement',
    'PhoreticAttractionInteraction',
    'PolarAlignmentInteraction',
    'PolarFluidElement',
    'TumblerElement',
    'TumblerView',
    'VicsekParticleElement',
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
