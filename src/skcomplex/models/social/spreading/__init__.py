"""Spreading: rumours, cascades, and the growth of texts."""

from ._cascades import ThresholdNodeElement, watts_cascades
from ._growing_word_view import GrowingWordView
from ._growth import GrowingWordElement, WritingInteraction, simon_yule
from ._rumours import ListenersElement, RumourContactsInteraction, maki_thompson
from ._threshold_node_view import ThresholdNodeView

__all__: list[str] = [
    'GrowingWordElement',
    'GrowingWordView',
    'ListenersElement',
    'RumourContactsInteraction',
    'ThresholdNodeElement',
    'ThresholdNodeView',
    'WritingInteraction',
    'maki_thompson',
    'simon_yule',
    'watts_cascades',
]
