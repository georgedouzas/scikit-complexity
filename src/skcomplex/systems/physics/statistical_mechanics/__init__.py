"""Statistical mechanics systems.

The package covers gases, fluids, spin systems, lattice processes, urns, master equations, and their measures.
"""

from ..common.spins import IsingSpinElement
from ._absorbing_processes import AbsorbingBirthDeathSystem, ReactingParticlesElement
from ._driven_particles import (
    ASEPSystem,
    DissipativeZeroRangeSystem,
    ExclusionHopsInteraction,
    ParticleSiteElement,
    ReservoirInteraction,
    TreeNodeElement,
    ZeroRangeHopInteraction,
    ZeroRangeHopsInteraction,
    ZeroRangeProcessSystem,
)
from ._fully_connected_spins import CurieWeissSystem, FullyConnectedSpinElement, SherringtonKirkpatrickSystem
from ._gases import IdealGasSystem, LennardJonesFluidSystem
from ._graph_spins import GraphIsingSystem
from ._kinetic_temperature import KineticTemperatureResult, kinetic_temperature
from ._master_equations import (
    DiscreteMasterEquationSystem,
    LandauRelaxationSystem,
    MicrostateElement,
    TwoPhaseZoneExchangeSystem,
    ZoneElement,
)
from ._maxwell_boltzmann import MaxwellBoltzmannResult, maxwell_boltzmann
from ._percolation import ContinuumPercolationSystem, DiskElement
from ._processes import (
    ContactProcessSystem,
    ContactSiteElement,
    DiffusionLimitedAggregationSystem,
    ExclusionSiteElement,
    PercolationSiteElement,
    Q2RSpinElement,
    Q2rSystem,
    SitePercolationSystem,
    TASEPSystem,
    WalkerElement,
    WalkerView,
)
from ._quantum_spins import (
    DephasingInteraction,
    QuantumHeisenbergSystem,
    XXZExchangeInteraction,
    ZeemanFieldInteraction,
    local_magnetization,
    total_magnetization,
)
from ._random_energy import (
    BoltzmannAverageElement,
    BoltzmannSumInteraction,
    EnergyLevelElement,
    RandomEnergySystem,
)
from ._spin_glasses import (
    BondSamplingInteraction,
    BondSpinElement,
    EdwardsAndersonSystem,
    MattisSpinElement,
    MattisSpinGlassSystem,
)
from ._spins import IsingSystem, PottsSpinElement, PottsSystem, XYSpinElement, XYSystem
from ._susceptibility import SusceptibilityResult, susceptibility
from ._traps import TrapSystem, TrapWalkerElement
from ._urns import EhrenfestUrnElement, EhrenfestUrnSystem

__all__: list[str] = [
    'ASEPSystem',
    'AbsorbingBirthDeathSystem',
    'BoltzmannAverageElement',
    'BoltzmannSumInteraction',
    'BondSamplingInteraction',
    'BondSpinElement',
    'ContactProcessSystem',
    'ContactSiteElement',
    'ContinuumPercolationSystem',
    'CurieWeissSystem',
    'DephasingInteraction',
    'DiffusionLimitedAggregationSystem',
    'DiscreteMasterEquationSystem',
    'DiskElement',
    'DissipativeZeroRangeSystem',
    'EdwardsAndersonSystem',
    'EhrenfestUrnElement',
    'EhrenfestUrnSystem',
    'EnergyLevelElement',
    'ExclusionHopsInteraction',
    'ExclusionSiteElement',
    'FullyConnectedSpinElement',
    'GraphIsingSystem',
    'IdealGasSystem',
    'IsingSpinElement',
    'IsingSystem',
    'KineticTemperatureResult',
    'LandauRelaxationSystem',
    'LennardJonesFluidSystem',
    'MattisSpinElement',
    'MattisSpinGlassSystem',
    'MaxwellBoltzmannResult',
    'MicrostateElement',
    'ParticleSiteElement',
    'PercolationSiteElement',
    'PottsSpinElement',
    'PottsSystem',
    'Q2RSpinElement',
    'Q2rSystem',
    'QuantumHeisenbergSystem',
    'RandomEnergySystem',
    'ReactingParticlesElement',
    'ReservoirInteraction',
    'SherringtonKirkpatrickSystem',
    'SitePercolationSystem',
    'SusceptibilityResult',
    'TASEPSystem',
    'TrapSystem',
    'TrapWalkerElement',
    'TreeNodeElement',
    'TwoPhaseZoneExchangeSystem',
    'WalkerElement',
    'WalkerView',
    'XXZExchangeInteraction',
    'XYSpinElement',
    'XYSystem',
    'ZeemanFieldInteraction',
    'ZeroRangeHopInteraction',
    'ZeroRangeHopsInteraction',
    'ZeroRangeProcessSystem',
    'ZoneElement',
    'kinetic_temperature',
    'local_magnetization',
    'maxwell_boltzmann',
    'susceptibility',
    'total_magnetization',
]
