"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, swarmalators, chaos, and Kuramoto order."""

from ._adaptive import AdaptiveCouplingInteraction, AdaptiveLinkElement, adaptive_kuramoto
from ._amplitude_death import (
    GlobalCouplingInteraction,
    heterogeneous_global_coupling,
    oscillator_death,
)
from ._chaos import (
    AggregateAveragingInteraction,
    DelayedUnitElement,
    DiffusiveCouplingInteraction,
    LogisticUnitElement,
    RosslerOscillatorElement,
    aggregate_averaging_synchronization,
    coupled_rossler,
    rossler_driven_lorenz,
)
from ._hypergraphs import SimplexCouplingInteraction, hypergraph_kuramoto
from ._kuramoto import (
    DelayedCouplingInteraction,
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)
from ._link import LinkElement
from ._networks import (
    EdgeCouplingInteraction,
    LogisticCouplingInteraction,
    LogisticNodeElement,
    coupled_map_lattices_on_networks,
    kuramoto_on_networks,
)
from ._pulses import IntegrateAndFireOscillatorElement, WinfreeMeanFieldInteraction, mirollo_strogatz, winfree
from ._swarmalators import SwarmalatorCouplingInteraction, SwarmalatorElement, swarmalators
from ._topological import topological_kuramoto

__all__: list[str] = [
    'AdaptiveCouplingInteraction',
    'AdaptiveLinkElement',
    'AggregateAveragingInteraction',
    'DelayedCouplingInteraction',
    'DelayedUnitElement',
    'DiffusiveCouplingInteraction',
    'EdgeCouplingInteraction',
    'GlobalCouplingInteraction',
    'IntegrateAndFireOscillatorElement',
    'LinkElement',
    'LogisticCouplingInteraction',
    'LogisticNodeElement',
    'LogisticUnitElement',
    'RosslerOscillatorElement',
    'SimplexCouplingInteraction',
    'SwarmalatorCouplingInteraction',
    'SwarmalatorElement',
    'WinfreeMeanFieldInteraction',
    'adaptive_kuramoto',
    'aggregate_averaging_synchronization',
    'coupled_map_lattices_on_networks',
    'coupled_rossler',
    'delayed_kuramoto',
    'heterogeneous_global_coupling',
    'hypergraph_kuramoto',
    'kuramoto',
    'kuramoto_on_networks',
    'kuramoto_sakaguchi_chimera',
    'mirollo_strogatz',
    'nonlocal_ring_chimera',
    'oscillator_death',
    'ott_antonsen',
    'rossler_driven_lorenz',
    'swarmalators',
    'topological_kuramoto',
    'winfree',
]
