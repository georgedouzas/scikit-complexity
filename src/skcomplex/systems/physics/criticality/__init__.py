"""Criticality: sandpiles, forest fires, absorbing-state transitions, learners at the edge, and avalanches."""

from ._absorbing import ActiveFieldElement, directed_percolation_langevin
from ._avalanche import AvalancheInteraction
from ._branching_tree import BranchingTreeElement
from ._crackling_noise import CracklingNoiseAnalysis
from ._criticality import forest_fire, hexagonal_forest_fire, manna_sandpile, sandpile
from ._energy_site import EnergySiteElement
from ._evolutionary_learning import LearnerElement, evolutionary_sk_game
from ._fire import FireInteraction
from ._forest import ForestElement
from ._learning import LearningAgentElement, sk_game
from ._sandpile_cell import SandpileCellElement
from ._sandpiles import fixed_energy_sandpile, self_organized_branching, zhang_sandpile
from ._topple import ToppleInteraction

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
