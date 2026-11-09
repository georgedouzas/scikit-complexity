"""Criticality systems.

The package covers sandpiles, forest fires, absorbing-state transitions, learners at the edge, and avalanches.
"""

from ..common import SandpileCellElement
from ._absorbing import ActiveFieldElement, DirectedPercolationLangevinSystem
from ._crackling_noise import CracklingNoiseResult, crackling_noise
from ._criticality import (
    AvalancheInteraction,
    FireInteraction,
    ForestElement,
    ForestFireSystem,
    HexagonalForestFireSystem,
    MannaSandpileSystem,
    SandpileSystem,
)
from ._evolutionary_learning import EvolutionarySKGameSystem, LearnerElement
from ._learning import LearningAgentElement, SKGameSystem
from ._sandpiles import (
    BranchingTreeElement,
    EnergySiteElement,
    FixedEnergySandpileSystem,
    SelfOrganizedBranchingSystem,
    ToppleInteraction,
    ZhangSandpileSystem,
)

__all__: list[str] = [
    'ActiveFieldElement',
    'AvalancheInteraction',
    'BranchingTreeElement',
    'CracklingNoiseResult',
    'DirectedPercolationLangevinSystem',
    'EnergySiteElement',
    'EvolutionarySKGameSystem',
    'FireInteraction',
    'FixedEnergySandpileSystem',
    'ForestElement',
    'ForestFireSystem',
    'HexagonalForestFireSystem',
    'LearnerElement',
    'LearningAgentElement',
    'MannaSandpileSystem',
    'SKGameSystem',
    'SandpileCellElement',
    'SandpileSystem',
    'SelfOrganizedBranchingSystem',
    'ToppleInteraction',
    'ZhangSandpileSystem',
    'crackling_noise',
]
