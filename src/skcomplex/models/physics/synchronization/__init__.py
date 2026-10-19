"""Synchronization: coupled phase oscillators, pulse-coupled oscillators, and synchronized chaos."""

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
from ._pulses import IntegrateAndFireOscillator, WinfreeMeanField, WinfreeOscillator, mirollo_strogatz, winfree

__all__: list[str] = [
    'AggregateAveraging',
    'DelayedCoupling',
    'DiffusiveCoupling',
    'IntegrateAndFireOscillator',
    'LogisticUnit',
    'MeanFieldAmplitude',
    'MeanFieldCoupling',
    'PhaseCoupling',
    'PhaseLag',
    'PhaseOscillator',
    'RosslerOscillator',
    'SitedOscillator',
    'WinfreeMeanField',
    'WinfreeOscillator',
    'aggregate_averaging_synchronization',
    'coupled_rossler',
    'delayed_kuramoto',
    'kuramoto',
    'kuramoto_sakaguchi_chimera',
    'mirollo_strogatz',
    'nonlocal_ring_chimera',
    'ott_antonsen',
    'winfree',
]
