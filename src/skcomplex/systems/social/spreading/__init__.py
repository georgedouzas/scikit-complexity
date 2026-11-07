"""Spreading: rumours, cascades, contagions on hypergraphs, and the growth of texts."""

from ._cascades import ThresholdActivationInteraction, ThresholdNodeElement, ThresholdNodeView, watts_cascades
from ._exposure import ExposureInteraction, ParticipantElement, hypergraph_exposure_sis
from ._growth import GrowingWordElement, GrowingWordView, WritingInteraction, simon_yule
from ._rumours import ListenersElement, RumourContactsInteraction, maki_thompson
from ._simplicial import InfectiveElement, SimplicialInfectionInteraction, simplicial_contagion

__all__: list[str] = [
    'ExposureInteraction',
    'GrowingWordElement',
    'GrowingWordView',
    'InfectiveElement',
    'ListenersElement',
    'ParticipantElement',
    'RumourContactsInteraction',
    'SimplicialInfectionInteraction',
    'ThresholdActivationInteraction',
    'ThresholdNodeElement',
    'ThresholdNodeView',
    'WritingInteraction',
    'hypergraph_exposure_sis',
    'maki_thompson',
    'simon_yule',
    'simplicial_contagion',
    'watts_cascades',
]
