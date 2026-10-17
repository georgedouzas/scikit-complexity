"""Compartmental epidemiology with deterministic and stochastic infection and recovery."""

from ._compartments import Compartment, Infection, Recovery

__all__: list[str] = [
    'Compartment',
    'Infection',
    'Recovery',
]
