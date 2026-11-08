"""Spreading: rumours, cascades, contagions on hypergraphs, and growing texts."""

from ._cascades import ThresholdActivationInteraction, ThresholdNodeElement, ThresholdNodeView, WattsCascadesSystem
from ._exposure import ExposureInteraction, HypergraphExposureSisSystem, ParticipantElement
from ._growth import GrowingWordElement, GrowingWordView, SimonYuleSystem, WritingInteraction
from ._rumours import ListenersElement, MakiThompsonSystem, RumourContactsInteraction
from ._simplicial import InfectiveElement, SimplicialContagionSystem, SimplicialInfectionInteraction

__all__: list[str] = [
    'ExposureInteraction',
    'GrowingWordElement',
    'GrowingWordView',
    'HypergraphExposureSisSystem',
    'InfectiveElement',
    'ListenersElement',
    'MakiThompsonSystem',
    'ParticipantElement',
    'RumourContactsInteraction',
    'SimonYuleSystem',
    'SimplicialContagionSystem',
    'SimplicialInfectionInteraction',
    'ThresholdActivationInteraction',
    'ThresholdNodeElement',
    'ThresholdNodeView',
    'WattsCascadesSystem',
    'WritingInteraction',
]
