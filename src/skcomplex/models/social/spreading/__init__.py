"""Spreading: rumours, cascades, and the growth of texts."""

from ._cascades import ThresholdNode, watts_cascades
from ._growth import Word, Writing, simon_yule
from ._rumours import Listeners, RumourContacts, maki_thompson

__all__: list[str] = [
    'Listeners',
    'RumourContacts',
    'ThresholdNode',
    'Word',
    'Writing',
    'maki_thompson',
    'simon_yule',
    'watts_cascades',
]
