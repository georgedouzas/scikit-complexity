"""Statistical mechanics: gases, fluids, spin models, lattice processes, urns, and master equations."""

from ._absorbing_processes import ReactingParticlesComponent, absorbing_birth_death
from ._driven_particles import (
    ExclusionHopsInteraction,
    ParticleSiteComponent,
    ReservoirInteraction,
    TreeNodeComponent,
    ZeroRangeHopInteraction,
    ZeroRangeHopsInteraction,
    asep,
    dissipative_zero_range,
    zero_range_process,
)
from ._fully_connected_spins import FullyConnectedSpinComponent, curie_weiss, sherrington_kirkpatrick
from ._gases import ideal_gas, lennard_jones_fluid
from ._graph_spins import GraphIsingSpinComponent, graph_ising
from ._master_equations import (
    MicrostateComponent,
    OrderParameterComponent,
    ZoneComponent,
    discrete_master_equation,
    landau_relaxation,
    two_phase_zone_exchange,
)
from ._processes import (
    ContactSiteComponent,
    ExclusionSiteComponent,
    PercolationSiteComponent,
    Q2RSpinComponent,
    WalkerComponent,
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
from ._spin_glasses import BondSpinComponent, MattisSpinComponent, edwards_anderson, mattis_spin_glass
from ._spins import IsingSpinComponent, PottsSpinComponent, XYSpinComponent, ising, potts, xy
from ._traps import TrapWalkerComponent, trap_model
from ._urns import EhrenfestUrnComponent, ehrenfest_urn

__all__: list[str] = [
    'BondSpinComponent',
    'ContactSiteComponent',
    'DephasingInteraction',
    'EhrenfestUrnComponent',
    'ExclusionHopsInteraction',
    'ExclusionSiteComponent',
    'FullyConnectedSpinComponent',
    'GraphIsingSpinComponent',
    'IsingSpinComponent',
    'MattisSpinComponent',
    'MicrostateComponent',
    'OrderParameterComponent',
    'ParticleSiteComponent',
    'PercolationSiteComponent',
    'PottsSpinComponent',
    'Q2RSpinComponent',
    'ReactingParticlesComponent',
    'ReservoirInteraction',
    'TrapWalkerComponent',
    'TreeNodeComponent',
    'WalkerComponent',
    'XXZExchangeInteraction',
    'XYSpinComponent',
    'ZeemanFieldInteraction',
    'ZeroRangeHopInteraction',
    'ZeroRangeHopsInteraction',
    'ZoneComponent',
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
