"""Active matter: self-propelled particles, flocking, polar fluids, chemotaxis, Active Model B+, and their orders."""

from ._active_particle import ActiveParticleElement
from ._alignment import PolarAlignmentInteraction, aligning_active_brownian_particles
from ._binary_collisions import BinaryCollisionsInteraction
from ._chemotaxis import ChemotaxisElement, keller_segel
from ._cohesive_particle import CohesiveParticleElement
from ._collective import cucker_smale_flocking, living_crystals, motility_induced_phase_separation, toner_tu
from ._collisions import binary_collision_flocking
from ._cucker_smale_alignment import CuckerSmaleAlignmentInteraction
from ._depot_particle import DepotParticleElement
from ._flocker import FlockerElement
from ._flocker_view import FlockerView
from ._flocks import ginelli_nematics, gregoire_chate, vicsek
from ._heading_alignment import HeadingAlignmentInteraction
from ._model_b import ActiveScalarElement, active_model_b
from ._nematic_particle import NematicParticleElement
from ._nonreciprocal import NonreciprocalMoverElement, nonreciprocal_vicsek
from ._ornstein_uhlenbeck_particle import OrnsteinUhlenbeckParticleElement
from ._particles import (
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    run_and_tumble,
)
from ._phoretic_attraction import PhoreticAttractionInteraction
from ._polar_fluid import PolarFluidElement
from ._propulsion_step import PropulsionStepInteraction
from ._self_propulsion import SelfPropulsionInteraction
from ._tumbler import TumblerElement
from ._tumbler_view import TumblerView
from ._vicsek_particle import VicsekParticleElement

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
    'PropulsionStepInteraction',
    'SelfPropulsionInteraction',
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
