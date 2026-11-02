"""Seismology: earthquakes on loaded faults."""

from ._faults import FaultElement, earthquakes
from ._spring_blocks import SpringBlockElement, burridge_knopoff

__all__: list[str] = [
    'FaultElement',
    'SpringBlockElement',
    'burridge_knopoff',
    'earthquakes',
]
