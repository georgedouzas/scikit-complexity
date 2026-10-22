"""Criticality: sandpiles, forest fires, absorbing-state transitions, and learning agents on a spin glass."""

from ._absorbing import ActiveField, directed_percolation_langevin
from ._criticality import (
    Forest,
    MannaCell,
    SandpileCell,
    SpreadingForest,
    forest_fire,
    hexagonal_forest_fire,
    manna_sandpile,
    sandpile,
)
from ._learning import LearningAgent, sk_game

__all__: list[str] = [
    'ActiveField',
    'Forest',
    'LearningAgent',
    'MannaCell',
    'SandpileCell',
    'SpreadingForest',
    'directed_percolation_langevin',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
    'sk_game',
]
