"""Segregation: residential sorting of agents."""

from ._schelling import Resident, schelling

__all__: list[str] = [
    'Resident',
    'schelling',
]
