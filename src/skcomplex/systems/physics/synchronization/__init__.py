"""Synchronization systems.

The package covers coupled phase oscillators, pulse-coupled oscillators, swarmalators, chaos, and Kuramoto order.
"""

from ..common.oscillators import LinkElement
from ._adaptive import AdaptiveCouplingInteraction, AdaptiveKuramotoSystem, AdaptiveLinkElement
from ._amplitude_death import (
    GlobalCouplingInteraction,
    HeterogeneousGlobalCouplingSystem,
    OscillatorDeathSystem,
)
from ._chaos import (
    AggregateAveragingInteraction,
    AggregateAveragingSynchronizationSystem,
    CoupledRosslerSystem,
    DelayedUnitElement,
    DiffusiveCouplingInteraction,
    LogisticUnitElement,
    RosslerDrivenLorenzSystem,
    RosslerOscillatorElement,
)
from ._hypergraphs import HypergraphKuramotoSystem, SimplexCouplingInteraction
from ._kuramoto import (
    DelayedCouplingInteraction,
    DelayedKuramotoSystem,
    KuramotoSakaguchiChimeraSystem,
    KuramotoSystem,
    NonlocalRingChimeraSystem,
    OttAntonsenSystem,
)
from ._networks import (
    CoupledMapLatticesOnNetworksSystem,
    EdgeCouplingInteraction,
    KuramotoOnNetworksSystem,
    LogisticCouplingInteraction,
    LogisticNodeElement,
)
from ._pulses import (
    IntegrateAndFireOscillatorElement,
    MirolloStrogatzSystem,
    WinfreeMeanFieldInteraction,
    WinfreeSystem,
)
from ._swarmalators import SwarmalatorCouplingInteraction, SwarmalatorElement, SwarmalatorsSystem
from ._topological import TopologicalKuramotoSystem

__all__: list[str] = [
    "AdaptiveCouplingInteraction",
    "AdaptiveKuramotoSystem",
    "AdaptiveLinkElement",
    "AggregateAveragingInteraction",
    "AggregateAveragingSynchronizationSystem",
    "CoupledMapLatticesOnNetworksSystem",
    "CoupledRosslerSystem",
    "DelayedCouplingInteraction",
    "DelayedKuramotoSystem",
    "DelayedUnitElement",
    "DiffusiveCouplingInteraction",
    "EdgeCouplingInteraction",
    "GlobalCouplingInteraction",
    "HeterogeneousGlobalCouplingSystem",
    "HypergraphKuramotoSystem",
    "IntegrateAndFireOscillatorElement",
    "KuramotoOnNetworksSystem",
    "KuramotoSakaguchiChimeraSystem",
    "KuramotoSystem",
    "LinkElement",
    "LogisticCouplingInteraction",
    "LogisticNodeElement",
    "LogisticUnitElement",
    "MirolloStrogatzSystem",
    "NonlocalRingChimeraSystem",
    "OscillatorDeathSystem",
    "OttAntonsenSystem",
    "RosslerDrivenLorenzSystem",
    "RosslerOscillatorElement",
    "SimplexCouplingInteraction",
    "SwarmalatorCouplingInteraction",
    "SwarmalatorElement",
    "SwarmalatorsSystem",
    "TopologicalKuramotoSystem",
    "WinfreeMeanFieldInteraction",
    "WinfreeSystem",
]
