"""Dynamical systems.

The package covers chaotic and driven flows, maps, Koopman embeddings, and normal forms of bifurcations.
"""

from ..common.flows._rossler import RosslerElement
from ._chaotic_ensembles import (
    CubicChaoticMapSystem,
    CubicWalkElement,
    DoublingMapElement,
    DoublingMapSystem,
    SkewTentMapElement,
    SkewTentMapSystem,
)
from ._driven_flows import DrivenRosslerElement, OUDrivenRosslerSystem
from ._flows import (
    FiniteTimeBlowUpSystem,
    HyperchaoticRosslerSystem,
    IkedaSystem,
    LangfordElement,
    LangfordSystem,
    LorenzSystem,
    MixedFlowElement,
    NoseHooverElement,
    NoseHooverSystem,
    RosslerSystem,
    SprottJerkSystem,
    SprottMixedFlowSystem,
    ThomasSystem,
)
from ._koopman import KoopmanSlowManifoldSystem
from ._maps import (
    BakerMapSystem,
    CatMapSystem,
    CDNMapSystem,
    CircleMapSystem,
    ConservativeHenonMapSystem,
    FermiUlamMapSystem,
    HenonMapSystem,
    IntermittencyMapSystem,
    LogisticMapElement,
    LogisticMapSystem,
    LoziMapSystem,
    NewtonRaphsonMapSystem,
    StandardMapSystem,
    TentMapSystem,
    TowelMapSystem,
)
from ._normal_forms import (
    CuspFormSystem,
    HopfFormSystem,
    SaddleNodeFormSystem,
    TakensBogdanovFormSystem,
)
from ._subcritical_forms import SubcriticalHopfElement, SubcriticalHopfFormSystem

__all__: list[str] = [
    'BakerMapSystem',
    'CDNMapSystem',
    'CatMapSystem',
    'CircleMapSystem',
    'ConservativeHenonMapSystem',
    'CubicChaoticMapSystem',
    'CubicWalkElement',
    'CuspFormSystem',
    'DoublingMapElement',
    'DoublingMapSystem',
    'DrivenRosslerElement',
    'FermiUlamMapSystem',
    'FiniteTimeBlowUpSystem',
    'HenonMapSystem',
    'HopfFormSystem',
    'HyperchaoticRosslerSystem',
    'IkedaSystem',
    'IntermittencyMapSystem',
    'KoopmanSlowManifoldSystem',
    'LangfordElement',
    'LangfordSystem',
    'LogisticMapElement',
    'LogisticMapSystem',
    'LorenzSystem',
    'LoziMapSystem',
    'MixedFlowElement',
    'NewtonRaphsonMapSystem',
    'NoseHooverElement',
    'NoseHooverSystem',
    'OUDrivenRosslerSystem',
    'RosslerElement',
    'RosslerSystem',
    'SaddleNodeFormSystem',
    'SkewTentMapElement',
    'SkewTentMapSystem',
    'SprottJerkSystem',
    'SprottMixedFlowSystem',
    'StandardMapSystem',
    'SubcriticalHopfElement',
    'SubcriticalHopfFormSystem',
    'TakensBogdanovFormSystem',
    'TentMapSystem',
    'ThomasSystem',
    'TowelMapSystem',
]
