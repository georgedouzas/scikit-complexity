"""Criticality: sandpiles, forest fires, absorbing-state transitions, and learners at the edge of ergodicity."""

from ._absorbing import ActiveFieldComponent, directed_percolation_langevin
from ._criticality import (
    ForestComponent,
    MannaCellComponent,
    SandpileCellComponent,
    SpreadingForestComponent,
    forest_fire,
    hexagonal_forest_fire,
    manna_sandpile,
    sandpile,
)
from ._evolutionary_learning import LearnerComponent, evolutionary_sk_game
from ._learning import LearningAgentComponent, sk_game

__all__: list[str] = [
    'ActiveFieldComponent',
    'ForestComponent',
    'LearnerComponent',
    'LearningAgentComponent',
    'MannaCellComponent',
    'SandpileCellComponent',
    'SpreadingForestComponent',
    'directed_percolation_langevin',
    'evolutionary_sk_game',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
    'sk_game',
]
