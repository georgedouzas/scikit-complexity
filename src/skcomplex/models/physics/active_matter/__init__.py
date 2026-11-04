"""Active matter: self-propelled particles, flocking, polar fluids, chemotaxis, Active Model B+, and their orders."""

from ._active_particle import ActiveParticleElement
from ._alignment import PolarAlignmentInteraction, aligning_active_brownian_particles
from ._angular_alignment import AngularAlignmentInteraction
from ._angular_diffusion import AngularDiffusionInteraction
from ._binary_collisions import BinaryCollisionsInteraction
from ._cells import MotileCellElement, szabo_cells
from ._chemotaxis import ChemotaxisElement, keller_segel
from ._cohesive_particle import CohesiveParticleElement
from ._collective import cucker_smale_flocking, living_crystals, motility_induced_phase_separation, toner_tu
from ._collisions import binary_collision_flocking
from ._contractile_gel_site import ContractileGelSiteElement
from ._cucker_smale_alignment import CuckerSmaleAlignmentInteraction
from ._depot_particle import DepotParticleElement
from ._flocker import FlockerElement
from ._flocker_view import FlockerView
from ._flocks import ginelli_nematics, gregoire_chate, vicsek
from ._gels import active_viscoelastic_gel
from ._heading_alignment import HeadingAlignmentInteraction
from ._kinetic_flocks import bertin_droz_gregoire
from ._landau_de_gennes_energy import LandauDeGennesEnergyInteraction
from ._model_b import ActiveScalarElement, active_model_b
from ._nematic_density_site import NematicDensitySiteElement
from ._nematic_gel_site import NematicGelSiteElement
from ._nematic_particle import NematicParticleElement
from ._nematics import active_gels, dry_active_nematics
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
from ._polar_gel_site import PolarGelSiteElement
from ._propulsion_step import PropulsionStepInteraction
from ._self_propulsion import SelfPropulsionInteraction
from ._tumbler import TumblerElement
from ._tumbler_view import TumblerView
from ._vicsek_particle import VicsekParticleElement

__all__: list[str] = [
    'ActiveParticleElement',
    'ActiveScalarElement',
    'AngularAlignmentInteraction',
    'AngularDiffusionInteraction',
    'BinaryCollisionsInteraction',
    'ChemotaxisElement',
    'CohesiveParticleElement',
    'ContractileGelSiteElement',
    'CuckerSmaleAlignmentInteraction',
    'DepotParticleElement',
    'FlockerElement',
    'FlockerView',
    'HeadingAlignmentInteraction',
    'LandauDeGennesEnergyInteraction',
    'MotileCellElement',
    'NematicDensitySiteElement',
    'NematicGelSiteElement',
    'NematicParticleElement',
    'NonreciprocalMoverElement',
    'OrnsteinUhlenbeckParticleElement',
    'PhoreticAttractionInteraction',
    'PolarAlignmentInteraction',
    'PolarFluidElement',
    'PolarGelSiteElement',
    'PropulsionStepInteraction',
    'SelfPropulsionInteraction',
    'TumblerElement',
    'TumblerView',
    'VicsekParticleElement',
    'active_brownian_particles',
    'active_gels',
    'active_model_b',
    'active_ornstein_uhlenbeck_particles',
    'active_viscoelastic_gel',
    'aligning_active_brownian_particles',
    'bertin_droz_gregoire',
    'binary_collision_flocking',
    'chiral_active_particles',
    'cucker_smale_flocking',
    'depot_brownian_particles',
    'dry_active_nematics',
    'ginelli_nematics',
    'gregoire_chate',
    'keller_segel',
    'living_crystals',
    'motility_induced_phase_separation',
    'nonreciprocal_vicsek',
    'run_and_tumble',
    'szabo_cells',
    'toner_tu',
    'vicsek',
]
