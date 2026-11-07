"""Active matter: self-propelled particles, flocking, polar fluids, chemotaxis, squirmers, and Active System B+."""

from ._active_particle import ActiveParticleElement
from ._active_passive import FlockingAgentElement, active_passive_flocking
from ._alignment import PolarAlignmentInteraction, aligning_active_brownian_particles
from ._cells import MotileCellElement, szabo_cells
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
from ._collisions import BinaryCollisionsInteraction, FlockerElement, FlockerView, binary_collision_flocking
from ._deformable import DeformableParticleElement, deformable_active_particles
from ._flocks import (
    CohesiveParticleElement,
    DelayedVicsekParticleElement,
    HeadingAlignmentInteraction,
    NematicParticleElement,
    VicsekParticleElement,
    ginelli_nematics,
    gregoire_chate,
    vicsek,
)
from ._gels import ContractileGelSiteElement, active_viscoelastic_gel
from ._kinetic_flocks import AngularAlignmentInteraction, AngularDiffusionInteraction, bertin_droz_gregoire
from ._nematics import (
    LandauDeGennesEnergyInteraction,
    NematicDensitySiteElement,
    NematicGelSiteElement,
    PolarGelSiteElement,
    active_gels,
    dry_active_nematics,
)
from ._nonreciprocal import NonreciprocalMoverElement, nonreciprocal_vicsek
from ._particles import (
    DepotParticleElement,
    InertialActiveParticleElement,
    OrnsteinUhlenbeckParticleElement,
    TumblerElement,
    TumblerView,
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    depot_brownian_particles,
    inertial_active_brownian_particles,
    run_and_tumble,
)
from ._polymers import ActiveBeadElement, active_polymer
from ._propulsion_step import PropulsionStepInteraction
from ._self_propulsion import SelfPropulsionInteraction
from ._squirmers import SquirmerElement, SquirmerSlipInteraction, squirmers
from ._system_b import ActiveScalarElement, active_system_b

__all__: list[str] = [
    'ActiveBeadElement',
    'ActiveParticleElement',
    'ActiveScalarElement',
    'AngularAlignmentInteraction',
    'AngularDiffusionInteraction',
    'BinaryCollisionsInteraction',
    'ChemotaxisElement',
    'CohesiveParticleElement',
    'ContractileGelSiteElement',
    'CuckerSmaleAlignmentInteraction',
    'DeformableParticleElement',
    'DelayedVicsekParticleElement',
    'DepotParticleElement',
    'FlockerElement',
    'FlockerView',
    'FlockingAgentElement',
    'HeadingAlignmentInteraction',
    'InertialActiveParticleElement',
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
    'SquirmerElement',
    'SquirmerSlipInteraction',
    'TumblerElement',
    'TumblerView',
    'VicsekParticleElement',
    'active_brownian_particles',
    'active_gels',
    'active_ornstein_uhlenbeck_particles',
    'active_passive_flocking',
    'active_polymer',
    'active_system_b',
    'active_viscoelastic_gel',
    'aligning_active_brownian_particles',
    'bertin_droz_gregoire',
    'binary_collision_flocking',
    'chiral_active_particles',
    'cucker_smale_flocking',
    'deformable_active_particles',
    'depot_brownian_particles',
    'dry_active_nematics',
    'ginelli_nematics',
    'gregoire_chate',
    'inertial_active_brownian_particles',
    'keller_segel',
    'living_crystals',
    'motility_induced_phase_separation',
    'nonreciprocal_vicsek',
    'run_and_tumble',
    'squirmers',
    'szabo_cells',
    'toner_tu',
    'vicsek',
]
