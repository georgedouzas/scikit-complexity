"""Active matter systems.

The package covers self-propelled particles, flocking, polar fluids, chemotaxis, squirmers, and Active System B+.
"""

from ..common.particles import ActiveParticleElement, PropulsionStepInteraction, SelfPropulsionInteraction
from ._active_passive import ActivePassiveFlockingSystem, FlockingAgentElement
from ._alignment import AligningActiveBrownianParticlesSystem, PolarAlignmentInteraction
from ._cells import MotileCellElement, SzaboCellsSystem
from ._chemotaxis import ChemotaxisElement, KellerSegelSystem
from ._collective import (
    CuckerSmaleAlignmentInteraction,
    CuckerSmaleFlockingSystem,
    LivingCrystalsSystem,
    MotilityInducedPhaseSeparationSystem,
    PhoreticAttractionInteraction,
    PolarFluidElement,
    TonerTuSystem,
)
from ._collisions import BinaryCollisionFlockingSystem, BinaryCollisionsInteraction, FlockerElement, FlockerView
from ._deformable import DeformableActiveParticlesSystem, DeformableParticleElement
from ._flocks import (
    CohesiveParticleElement,
    DelayedVicsekParticleElement,
    GinelliNematicsSystem,
    GregoireChateSystem,
    HeadingAlignmentInteraction,
    NematicParticleElement,
    VicsekParticleElement,
    VicsekSystem,
)
from ._gels import ActiveViscoelasticGelSystem, ContractileGelSiteElement
from ._kinetic_flocks import AngularAlignmentInteraction, AngularDiffusionInteraction, BertinDrozGregoireSystem
from ._nematics import (
    ActiveGelsSystem,
    DryActiveNematicsSystem,
    LandauDeGennesEnergyInteraction,
    NematicDensitySiteElement,
    NematicGelSiteElement,
    PolarGelSiteElement,
)
from ._nonreciprocal import NonreciprocalMoverElement, NonreciprocalVicsekSystem
from ._particles import (
    ActiveBrownianParticlesSystem,
    ActiveOrnsteinUhlenbeckParticlesSystem,
    ChiralActiveParticlesSystem,
    DepotBrownianParticlesSystem,
    DepotParticleElement,
    InertialActiveBrownianParticlesSystem,
    InertialActiveParticleElement,
    OrnsteinUhlenbeckParticleElement,
    RunAndTumbleSystem,
    TumblerElement,
    TumblerView,
)
from ._polymers import ActiveBeadElement, ActivePolymerSystem
from ._squirmers import SquirmerElement, SquirmerSlipInteraction, SquirmersSystem
from ._system_b import ActiveScalarElement, ActiveSystemBSystem

__all__: list[str] = [
    'ActiveBeadElement',
    'ActiveBrownianParticlesSystem',
    'ActiveGelsSystem',
    'ActiveOrnsteinUhlenbeckParticlesSystem',
    'ActiveParticleElement',
    'ActivePassiveFlockingSystem',
    'ActivePolymerSystem',
    'ActiveScalarElement',
    'ActiveSystemBSystem',
    'ActiveViscoelasticGelSystem',
    'AligningActiveBrownianParticlesSystem',
    'AngularAlignmentInteraction',
    'AngularDiffusionInteraction',
    'BertinDrozGregoireSystem',
    'BinaryCollisionFlockingSystem',
    'BinaryCollisionsInteraction',
    'ChemotaxisElement',
    'ChiralActiveParticlesSystem',
    'CohesiveParticleElement',
    'ContractileGelSiteElement',
    'CuckerSmaleAlignmentInteraction',
    'CuckerSmaleFlockingSystem',
    'DeformableActiveParticlesSystem',
    'DeformableParticleElement',
    'DelayedVicsekParticleElement',
    'DepotBrownianParticlesSystem',
    'DepotParticleElement',
    'DryActiveNematicsSystem',
    'FlockerElement',
    'FlockerView',
    'FlockingAgentElement',
    'GinelliNematicsSystem',
    'GregoireChateSystem',
    'HeadingAlignmentInteraction',
    'InertialActiveBrownianParticlesSystem',
    'InertialActiveParticleElement',
    'KellerSegelSystem',
    'LandauDeGennesEnergyInteraction',
    'LivingCrystalsSystem',
    'MotileCellElement',
    'MotilityInducedPhaseSeparationSystem',
    'NematicDensitySiteElement',
    'NematicGelSiteElement',
    'NematicParticleElement',
    'NonreciprocalMoverElement',
    'NonreciprocalVicsekSystem',
    'OrnsteinUhlenbeckParticleElement',
    'PhoreticAttractionInteraction',
    'PolarAlignmentInteraction',
    'PolarFluidElement',
    'PolarGelSiteElement',
    'PropulsionStepInteraction',
    'RunAndTumbleSystem',
    'SelfPropulsionInteraction',
    'SquirmerElement',
    'SquirmerSlipInteraction',
    'SquirmersSystem',
    'SzaboCellsSystem',
    'TonerTuSystem',
    'TumblerElement',
    'TumblerView',
    'VicsekParticleElement',
    'VicsekSystem',
]
