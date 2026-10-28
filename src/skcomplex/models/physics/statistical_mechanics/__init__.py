"""Statistical mechanics: gases, fluids, spin models, lattice processes, urns, master equations, and their measures."""

from ._absorbing_processes import ReactingParticlesElement, absorbing_birth_death
from ._binder_cumulant import BinderCumulantAnalysis
from ._critical_exponent import CriticalExponentAnalysis
from ._driven_particles import (
    ExclusionHopsInteraction,
    ParticleSiteElement,
    ReservoirInteraction,
    TreeNodeElement,
    ZeroRangeHopInteraction,
    ZeroRangeHopsInteraction,
    asep,
    dissipative_zero_range,
    zero_range_process,
)
from ._fully_connected_spins import FullyConnectedSpinElement, curie_weiss, sherrington_kirkpatrick
from ._gases import ideal_gas, lennard_jones_fluid
from ._graph_spins import GraphIsingSpinElement, graph_ising
from ._kinetic_temperature import KineticTemperatureAnalysis
from ._master_equations import (
    MicrostateElement,
    OrderParameterElement,
    ZoneElement,
    discrete_master_equation,
    landau_relaxation,
    two_phase_zone_exchange,
)
from ._maxwell_boltzmann import MaxwellBoltzmannAnalysis
from ._processes import (
    ContactSiteElement,
    ExclusionSiteElement,
    PercolationSiteElement,
    Q2RSpinElement,
    WalkerElement,
    contact_process,
    diffusion_limited_aggregation,
    q2r,
    site_percolation,
    tasep,
)
from ._quantum_spins import (
    DephasingInteraction,
    XXZExchangeInteraction,
    ZeemanFieldInteraction,
    local_magnetization,
    quantum_heisenberg,
    total_magnetization,
)
from ._replica_overlap import ReplicaOverlapAnalysis
from ._spin_glasses import BondSpinElement, MattisSpinElement, edwards_anderson, mattis_spin_glass
from ._spins import IsingSpinElement, PottsSpinElement, XYSpinElement, ising, potts, xy
from ._susceptibility import SusceptibilityAnalysis
from ._traps import TrapWalkerElement, trap_model
from ._urns import EhrenfestUrnElement, ehrenfest_urn

__all__: list[str] = [
    'BinderCumulantAnalysis',
    'BondSpinElement',
    'ContactSiteElement',
    'CriticalExponentAnalysis',
    'DephasingInteraction',
    'EhrenfestUrnElement',
    'ExclusionHopsInteraction',
    'ExclusionSiteElement',
    'FullyConnectedSpinElement',
    'GraphIsingSpinElement',
    'IsingSpinElement',
    'KineticTemperatureAnalysis',
    'MattisSpinElement',
    'MaxwellBoltzmannAnalysis',
    'MicrostateElement',
    'OrderParameterElement',
    'ParticleSiteElement',
    'PercolationSiteElement',
    'PottsSpinElement',
    'Q2RSpinElement',
    'ReactingParticlesElement',
    'ReplicaOverlapAnalysis',
    'ReservoirInteraction',
    'SusceptibilityAnalysis',
    'TrapWalkerElement',
    'TreeNodeElement',
    'WalkerElement',
    'XXZExchangeInteraction',
    'XYSpinElement',
    'ZeemanFieldInteraction',
    'ZeroRangeHopInteraction',
    'ZeroRangeHopsInteraction',
    'ZoneElement',
    'absorbing_birth_death',
    'asep',
    'contact_process',
    'curie_weiss',
    'diffusion_limited_aggregation',
    'discrete_master_equation',
    'dissipative_zero_range',
    'edwards_anderson',
    'ehrenfest_urn',
    'graph_ising',
    'ideal_gas',
    'ising',
    'landau_relaxation',
    'lennard_jones_fluid',
    'local_magnetization',
    'mattis_spin_glass',
    'potts',
    'q2r',
    'quantum_heisenberg',
    'sherrington_kirkpatrick',
    'site_percolation',
    'tasep',
    'total_magnetization',
    'trap_model',
    'two_phase_zone_exchange',
    'xy',
    'zero_range_process',
]
