"""Epidemiology: compartmental models of infection and recovery."""

from ._compartments import Compartment, Infection, Recovery
from ._factories import sir

__all__: list[str] = [
    'Compartment',
    'Infection',
    'Recovery',
    'sir',
]
