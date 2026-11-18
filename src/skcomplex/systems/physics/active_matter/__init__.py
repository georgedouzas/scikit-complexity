"""Active matter systems.

The package covers self-propelled particles, flocking, polar fluids, chemotaxis, squirmers, and Active System B+.
"""

from ..common.particles import (
    ActiveDiffusionInteraction,
    ActiveParticleElement,
    PropulsionStepInteraction,
    SelfPropulsionInteraction,
)
from ._active_passive import (
    ActivePassiveFlockingSystem,
    FlockingAgentElement,
    TargetSpeedInteraction,
    VelocityFlockingInteraction,
)
from ._alignment import AligningActiveBrownianParticlesSystem, PolarAlignmentInteraction
from ._cells import CellMotilityInteraction, MotileCellElement, SzaboCellsSystem
from ._chemotaxis import ChemotaxisElement, ChemotaxisInteraction, KellerSegelSystem
from ._collective import (
    BirdElement,
    CuckerSmaleAlignmentInteraction,
    CuckerSmaleFlockingSystem,
    LivingCrystalsSystem,
    MotilityInducedPhaseSeparationSystem,
    PhoreticAttractionInteraction,
    PolarFluidElement,
    TonerTuInteraction,
    TonerTuSystem,
)
from ._collisions import BinaryCollisionFlockingSystem, BinaryCollisionsInteraction, FlockerElement, FlockerView
from ._deformable import (
    DeformableActiveParticlesSystem,
    DeformableParticleElement,
    ShapeVelocityCouplingInteraction,
)
from ._flocks import (
    CohesiveAlignmentInteraction,
    CohesiveParticleElement,
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
from ._nonreciprocal import (
    NonreciprocalAlignmentInteraction,
    NonreciprocalMoverElement,
    NonreciprocalVicsekSystem,
)
from ._particles import (
    ActiveBrownianParticlesSystem,
    ActiveOrnsteinUhlenbeckParticlesSystem,
    ChiralActiveParticlesSystem,
    DepotBrownianParticlesSystem,
    DepotParticleElement,
    DepotPumpingInteraction,
    InertialActiveBrownianParticlesSystem,
    InertialActiveParticleElement,
    InertialPropulsionInteraction,
    OrnsteinUhlenbeckParticleElement,
    OrnsteinUhlenbeckPropulsionInteraction,
    RunAndTumbleInteraction,
    RunAndTumbleSystem,
    TumblerElement,
    TumblerView,
)
from ._polymers import ActiveBeadElement, ActivePolymerSystem, ActiveRouseInteraction
from ._squirmers import SquirmerElement, SquirmerSlipInteraction, SquirmersSystem
from ._system_b import ActiveCahnHilliardInteraction, ActiveScalarElement, ActiveSystemBSystem

__all__: list[str] = [
    'ActiveBeadElement',
    'ActiveBrownianParticlesSystem',
    'ActiveCahnHilliardInteraction',
    'ActiveDiffusionInteraction',
    'ActiveGelsSystem',
    'ActiveOrnsteinUhlenbeckParticlesSystem',
    'ActiveParticleElement',
    'ActivePassiveFlockingSystem',
    'ActivePolymerSystem',
    'ActiveRouseInteraction',
    'ActiveScalarElement',
    'ActiveSystemBSystem',
    'ActiveViscoelasticGelSystem',
    'AligningActiveBrownianParticlesSystem',
    'AngularAlignmentInteraction',
    'AngularDiffusionInteraction',
    'BertinDrozGregoireSystem',
    'BinaryCollisionFlockingSystem',
    'BinaryCollisionsInteraction',
    'BirdElement',
    'CellMotilityInteraction',
    'ChemotaxisElement',
    'ChemotaxisInteraction',
    'ChiralActiveParticlesSystem',
    'CohesiveAlignmentInteraction',
    'CohesiveParticleElement',
    'ContractileGelSiteElement',
    'CuckerSmaleAlignmentInteraction',
    'CuckerSmaleFlockingSystem',
    'DeformableActiveParticlesSystem',
    'DeformableParticleElement',
    'DepotBrownianParticlesSystem',
    'DepotParticleElement',
    'DepotPumpingInteraction',
    'DryActiveNematicsSystem',
    'FlockerElement',
    'FlockerView',
    'FlockingAgentElement',
    'GinelliNematicsSystem',
    'GregoireChateSystem',
    'HeadingAlignmentInteraction',
    'InertialActiveBrownianParticlesSystem',
    'InertialActiveParticleElement',
    'InertialPropulsionInteraction',
    'KellerSegelSystem',
    'LandauDeGennesEnergyInteraction',
    'LivingCrystalsSystem',
    'MotileCellElement',
    'MotilityInducedPhaseSeparationSystem',
    'NematicDensitySiteElement',
    'NematicGelSiteElement',
    'NematicParticleElement',
    'NonreciprocalAlignmentInteraction',
    'NonreciprocalMoverElement',
    'NonreciprocalVicsekSystem',
    'OrnsteinUhlenbeckParticleElement',
    'OrnsteinUhlenbeckPropulsionInteraction',
    'PhoreticAttractionInteraction',
    'PolarAlignmentInteraction',
    'PolarFluidElement',
    'PolarGelSiteElement',
    'PropulsionStepInteraction',
    'RunAndTumbleInteraction',
    'RunAndTumbleSystem',
    'SelfPropulsionInteraction',
    'ShapeVelocityCouplingInteraction',
    'SquirmerElement',
    'SquirmerSlipInteraction',
    'SquirmersSystem',
    'SzaboCellsSystem',
    'TargetSpeedInteraction',
    'TonerTuInteraction',
    'TonerTuSystem',
    'TumblerElement',
    'TumblerView',
    'VelocityFlockingInteraction',
    'VicsekParticleElement',
    'VicsekSystem',
]
