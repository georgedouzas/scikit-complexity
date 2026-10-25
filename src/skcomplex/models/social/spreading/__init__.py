"""Spreading: rumours, cascades, and the growth of texts."""

from ._cascades import ThresholdNodeComponent, watts_cascades
from ._growth import WordComponent, WritingInteraction, simon_yule
from ._rumours import ListenersComponent, RumourContactsInteraction, maki_thompson

__all__: list[str] = [
    'ListenersComponent',
    'RumourContactsInteraction',
    'ThresholdNodeComponent',
    'WordComponent',
    'WritingInteraction',
    'maki_thompson',
    'simon_yule',
    'watts_cascades',
]
