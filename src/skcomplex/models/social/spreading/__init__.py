"""Spreading: rumours, cascades, and the growth of texts."""

from ._cascades import ThresholdNodeElement, watts_cascades
from ._growth import WordElement, WritingInteraction, simon_yule
from ._rumours import ListenersElement, RumourContactsInteraction, maki_thompson

__all__: list[str] = [
    'ListenersElement',
    'RumourContactsInteraction',
    'ThresholdNodeElement',
    'WordElement',
    'WritingInteraction',
    'maki_thompson',
    'simon_yule',
    'watts_cascades',
]
