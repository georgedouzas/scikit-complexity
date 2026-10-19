"""Synchronization: coupled phase oscillators."""

from ._kuramoto import DelayedCoupling, PhaseOscillator, delayed_kuramoto

__all__: list[str] = [
    'DelayedCoupling',
    'PhaseOscillator',
    'delayed_kuramoto',
]
