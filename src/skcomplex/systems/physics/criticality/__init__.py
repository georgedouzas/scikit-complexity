"""Criticality: sandpiles, forest fires, absorbing-state transitions, learners at the edge, and avalanches."""

from ..common import SandpileCellElement
from ._absorbing import ActiveFieldElement, directed_percolation_langevin
from ._crackling_noise import CracklingNoiseAnalysis
from ._criticality import (
    AvalancheInteraction,
    FireInteraction,
    ForestElement,
    forest_fire,
    hexagonal_forest_fire,
    manna_sandpile,
    sandpile,
)
from ._evolutionary_learning import LearnerElement, evolutionary_sk_game
from ._learning import LearningAgentElement, sk_game
from ._sandpiles import (
    BranchingTreeElement,
    EnergySiteElement,
    ToppleInteraction,
    fixed_energy_sandpile,
    self_organized_branching,
    zhang_sandpile,
)

__all__: list[str] = [
    'ActiveFieldElement',
    'AvalancheInteraction',
    'BranchingTreeElement',
    'CracklingNoiseAnalysis',
    'EnergySiteElement',
    'FireInteraction',
    'ForestElement',
    'LearnerElement',
    'LearningAgentElement',
    'SandpileCellElement',
    'ToppleInteraction',
    'directed_percolation_langevin',
    'evolutionary_sk_game',
    'fixed_energy_sandpile',
    'forest_fire',
    'hexagonal_forest_fire',
    'manna_sandpile',
    'sandpile',
    'self_organized_branching',
    'sk_game',
    'zhang_sandpile',
]
