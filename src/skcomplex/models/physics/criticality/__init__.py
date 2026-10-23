"""Criticality: sandpiles, forest fires, absorbing-state transitions, and learners at the edge of ergodicity."""

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
from ._evolutionary_learning import Learner, evolutionary_sk_game
from ._learning import LearningAgent, sk_game

__all__: list[str] = [
    'ActiveField',
    'Forest',
    'Learner',
    'LearningAgent',
    'MannaCell',
    'SandpileCell',
    'SpreadingForest',
    'directed_percolation_langevin',
    'evolutionary_sk_game',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
    'sk_game',
]
