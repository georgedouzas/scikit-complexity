"""Oceanography systems.

The package covers the thermohaline circulation and the wind-driven gyres.
"""

from ._gyres import DoubleGyreSystem, QuasiGeostrophicSiteElement, WindStressInteraction
from ._thermohaline import StommelInteraction, StommelSystem

__all__: list[str] = [
    'DoubleGyreSystem',
    'QuasiGeostrophicSiteElement',
    'StommelInteraction',
    'StommelSystem',
    'WindStressInteraction',
]
