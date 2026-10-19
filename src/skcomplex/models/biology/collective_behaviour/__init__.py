"""Collective behaviour: animals that coordinate through signals."""

from ._foraging import Ant, Pheromone, pheromone_foraging

__all__: list[str] = [
    'Ant',
    'Pheromone',
    'pheromone_foraging',
]
