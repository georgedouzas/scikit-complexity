"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, swarmalators, and synchronized chaos."""

from ._amplitude_death import GlobalCoupling, StuartLandau, heterogeneous_global_coupling, oscillator_death
from ._chaos import (
    AggregateAveraging,
    DiffusiveCoupling,
    LogisticUnit,
    RosslerOscillator,
    aggregate_averaging_synchronization,
    coupled_rossler,
)
from ._kuramoto import (
    DelayedCoupling,
    MeanFieldAmplitude,
    MeanFieldCoupling,
    PhaseCoupling,
    PhaseLag,
    PhaseOscillator,
    SitedOscillator,
    delayed_kuramoto,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    nonlocal_ring_chimera,
    ott_antonsen,
)
from ._networks import EdgeCoupling, Link, LogisticNode, coupled_map_lattices_on_networks, kuramoto_on_networks
from ._pulses import IntegrateAndFireOscillator, WinfreeMeanField, WinfreeOscillator, mirollo_strogatz, winfree
from ._swarmalators import Swarmalator, SwarmalatorCoupling, swarmalators

__all__: list[str] = [
    'AggregateAveraging',
    'DelayedCoupling',
    'DiffusiveCoupling',
    'EdgeCoupling',
    'GlobalCoupling',
    'IntegrateAndFireOscillator',
    'Link',
    'LogisticNode',
    'LogisticUnit',
    'MeanFieldAmplitude',
    'MeanFieldCoupling',
    'PhaseCoupling',
    'PhaseLag',
    'PhaseOscillator',
    'RosslerOscillator',
    'SitedOscillator',
    'StuartLandau',
    'Swarmalator',
    'SwarmalatorCoupling',
    'WinfreeMeanField',
    'WinfreeOscillator',
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
