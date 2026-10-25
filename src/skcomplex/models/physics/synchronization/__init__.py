"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, swarmalators, and synchronized chaos."""

from ._amplitude_death import (
    GlobalCouplingInteraction,
    StuartLandauComponent,
    heterogeneous_global_coupling,
    oscillator_death,
)
from ._chaos import (
    AggregateAveragingInteraction,
    DiffusiveCouplingInteraction,
    LogisticUnitComponent,
    RosslerOscillatorComponent,
    aggregate_averaging_synchronization,
    coupled_rossler,
)
from ._kuramoto import (
    DelayedCouplingInteraction,
    MeanFieldAmplitudeComponent,
    MeanFieldCouplingInteraction,
    PhaseCouplingInteraction,
    PhaseLagInteraction,
    PhaseOscillatorComponent,
    SitedOscillatorComponent,
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)
from ._networks import (
    EdgeCouplingInteraction,
    LinkComponent,
    LogisticNodeComponent,
    coupled_map_lattices_on_networks,
    kuramoto_on_networks,
)
from ._pulses import (
    IntegrateAndFireOscillatorComponent,
    WinfreeMeanFieldInteraction,
    WinfreeOscillatorComponent,
    mirollo_strogatz,
    winfree,
)
from ._swarmalators import SwarmalatorComponent, SwarmalatorCouplingInteraction, swarmalators

__all__: list[str] = [
    'AggregateAveragingInteraction',
    'DelayedCouplingInteraction',
    'DiffusiveCouplingInteraction',
    'EdgeCouplingInteraction',
    'GlobalCouplingInteraction',
    'IntegrateAndFireOscillatorComponent',
    'LinkComponent',
    'LogisticNodeComponent',
    'LogisticUnitComponent',
    'MeanFieldAmplitudeComponent',
    'MeanFieldCouplingInteraction',
    'PhaseCouplingInteraction',
    'PhaseLagInteraction',
    'PhaseOscillatorComponent',
    'RosslerOscillatorComponent',
    'SitedOscillatorComponent',
    'StuartLandauComponent',
    'SwarmalatorComponent',
    'SwarmalatorCouplingInteraction',
    'WinfreeMeanFieldInteraction',
    'WinfreeOscillatorComponent',
    'aggregate_averaging_synchronization',
    'coupled_map_lattices_on_networks',
    'coupled_rossler',
    'delayed_kuramoto',
    'heterogeneous_global_coupling',
    'kuramoto',
    'kuramoto_on_networks',
    'kuramoto_sakaguchi_chimera',
    'mirollo_strogatz',
    'nonlocal_ring_chimera',
    'oscillator_death',
    'ott_antonsen',
    'swarmalators',
    'winfree',
]
