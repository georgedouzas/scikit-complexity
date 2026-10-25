"""Seismology: earthquakes on loaded faults."""

from ._faults import FaultComponent, earthquakes
from ._spring_blocks import SpringBlockComponent, burridge_knopoff

__all__: list[str] = [
    'FaultComponent',
    'SpringBlockComponent',
    'burridge_knopoff',
    'earthquakes',
]
