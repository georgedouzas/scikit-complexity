"""Seismology: earthquakes on loaded faults."""

from ._faults import Fault, earthquakes

__all__: list[str] = [
    'Fault',
    'earthquakes',
]
