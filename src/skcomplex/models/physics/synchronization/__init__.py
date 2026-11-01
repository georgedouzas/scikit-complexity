"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, swarmalators, chaos, and Kuramoto order."""

from ._amplitude_death import (
    GlobalCouplingInteraction,
    heterogeneous_global_coupling,
    oscillator_death,
)
from ._chaos import (
    AggregateAveragingInteraction,
    DiffusiveCouplingInteraction,
    LogisticUnitElement,
    RosslerOscillatorElement,
    aggregate_averaging_synchronization,
    coupled_rossler,
)
from ._kuramoto import (
    DelayedCouplingInteraction,
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)
from ._logistic_coupling import LogisticCouplingInteraction
from ._networks import (
    EdgeCouplingInteraction,
    LinkElement,
    LogisticNodeElement,
    coupled_map_lattices_on_networks,
    kuramoto_on_networks,
)
from ._pulses import (
    IntegrateAndFireOscillatorElement,
    WinfreeMeanFieldInteraction,
    mirollo_strogatz,
    winfree,
)
from ._swarmalators import SwarmalatorCouplingInteraction, swarmalators

__all__: list[str] = [
    'AggregateAveragingInteraction',
    'DelayedCouplingInteraction',
    'DiffusiveCouplingInteraction',
    'EdgeCouplingInteraction',
    'GlobalCouplingInteraction',
    'IntegrateAndFireOscillatorElement',
    'LinkElement',
    'LogisticCouplingInteraction',
    'LogisticNodeElement',
    'LogisticUnitElement',
    'RosslerOscillatorElement',
    'SwarmalatorCouplingInteraction',
    'WinfreeMeanFieldInteraction',
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
