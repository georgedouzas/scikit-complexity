"""Spreading: rumours, cascades, contagions on hypergraphs, and the growth of texts."""

from ._cascades import ThresholdNodeElement, watts_cascades
from ._exposure import ExposureInteraction, hypergraph_exposure_sis
from ._growing_word import GrowingWordElement
from ._growing_word_view import GrowingWordView
from ._growth import simon_yule
from ._infective import InfectiveElement
from ._listeners import ListenersElement
from ._participant import ParticipantElement
from ._rumour_contacts import RumourContactsInteraction
from ._rumours import maki_thompson
from ._simplicial import simplicial_contagion
from ._simplicial_infection import SimplicialInfectionInteraction
from ._threshold_activation import ThresholdActivationInteraction
from ._threshold_node_view import ThresholdNodeView
from ._writing import WritingInteraction

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
