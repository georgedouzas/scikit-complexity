"""Criticality: sandpiles, forest fires, absorbing-state transitions, learners at the edge, and avalanches."""

from ._absorbing import ActiveFieldElement, directed_percolation_langevin
from ._crackling_noise import CracklingNoiseAnalysis
from ._criticality import (
    ForestElement,
    MannaCellElement,
    SandpileCellElement,
    SpreadingForestElement,
    forest_fire,
    hexagonal_forest_fire,
    manna_sandpile,
    sandpile,
)
from ._evolutionary_learning import LearnerElement, evolutionary_sk_game
from ._learning import LearningAgentElement, sk_game

__all__: list[str] = [
    'ActiveFieldElement',
    'CracklingNoiseAnalysis',
    'ForestElement',
    'LearnerElement',
    'LearningAgentElement',
    'MannaCellElement',
    'SandpileCellElement',
    'SpreadingForestElement',
    'directed_percolation_langevin',
    'evolutionary_sk_game',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
    'sk_game',
]
