"""Criticality systems.

The package covers sandpiles, forest fires, absorbing-state transitions, learners at the edge, and avalanches.
"""

from ._absorbing import ActiveFieldElement, ActivePercolationInteraction, DirectedPercolationLangevinSystem
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
from ._evolutionary_learning import EvolutionarySKGameSystem, LearnerElement, SKLearningInteraction
from ._learning import LearningAgentElement, SKGameSystem, SKReinforcementInteraction
from ._sandpiles import (
    BranchingInteraction,
    EnergySiteElement,
    FixedEnergySandpileSystem,
    GrainSiteElement,
    SelfOrganizedBranchingSystem,
    ToppleInteraction,
    ZhangSandpileSystem,
)

__all__: list[str] = [
    'ActiveFieldElement',
    'ActivePercolationInteraction',
    'AvalancheInteraction',
    'BranchingInteraction',
    'CracklingNoiseResult',
    'DirectedPercolationLangevinSystem',
    'EnergySiteElement',
    'EvolutionarySKGameSystem',
    'FireInteraction',
    'FixedEnergySandpileSystem',
    'ForestElement',
    'ForestFireSystem',
    'GrainSiteElement',
    'HexagonalForestFireSystem',
    'LearnerElement',
    'LearningAgentElement',
    'MannaSandpileSystem',
    'SKGameSystem',
    'SKLearningInteraction',
    'SKReinforcementInteraction',
    'SandpileSystem',
    'SelfOrganizedBranchingSystem',
    'ToppleInteraction',
    'ZhangSandpileSystem',
    'crackling_noise',
]
