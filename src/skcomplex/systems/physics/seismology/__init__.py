"""Seismology systems.

The package covers earthquakes on loaded faults.
"""

from ._faults import EarthquakesSystem, FaultElement
from ._spring_blocks import BurridgeKnopoffSystem, SpringBlockElement

__all__: list[str] = [
    "BurridgeKnopoffSystem",
    "EarthquakesSystem",
    "FaultElement",
    "SpringBlockElement",
]
