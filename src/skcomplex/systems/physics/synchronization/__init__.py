"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, swarmalators, chaos, and Kuramoto order."""

from ._adaptive import adaptive_kuramoto
from ._adaptive_coupling import AdaptiveCouplingInteraction
from ._adaptive_link import AdaptiveLinkElement
from ._aggregate_averaging import AggregateAveragingInteraction
from ._amplitude_death import (
    GlobalCouplingInteraction,
    heterogeneous_global_coupling,
    oscillator_death,
)
from ._chaos import aggregate_averaging_synchronization, coupled_rossler, rossler_driven_lorenz
from ._delayed_unit import DelayedUnitElement
from ._diffusive_coupling import DiffusiveCouplingInteraction
from ._edge_coupling import EdgeCouplingInteraction
from ._hypergraphs import SimplexCouplingInteraction, hypergraph_kuramoto
from ._integrate_and_fire_oscillator import IntegrateAndFireOscillatorElement
from ._kuramoto import (
    DelayedCouplingInteraction,
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)
from ._link import LinkElement
from ._logistic_coupling import LogisticCouplingInteraction
from ._logistic_node import LogisticNodeElement
from ._logistic_unit import LogisticUnitElement
from ._networks import coupled_map_lattices_on_networks, kuramoto_on_networks
from ._pulses import mirollo_strogatz, winfree
from ._rossler_oscillator import RosslerOscillatorElement
from ._swarmalator import SwarmalatorElement
from ._swarmalators import SwarmalatorCouplingInteraction, swarmalators
from ._topological import topological_kuramoto
from ._winfree_mean_field import WinfreeMeanFieldInteraction

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
