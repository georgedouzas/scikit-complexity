"""Seismology: earthquakes on loaded faults."""

from ._faults import Fault, earthquakes
from ._spring_blocks import SpringBlock, burridge_knopoff

__all__: list[str] = [
    'Fault',
    'SpringBlock',
    'burridge_knopoff',
    'earthquakes',
]
