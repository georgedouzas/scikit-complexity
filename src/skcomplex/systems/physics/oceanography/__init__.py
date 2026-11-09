"""Oceanography systems.

The package covers the thermohaline circulation and the wind-driven gyres.
"""

from ._gyres import DoubleGyreSystem, QuasiGeostrophicSiteElement
from ._thermohaline import StommelBoxesElement, StommelSystem

__all__: list[str] = [
    "DoubleGyreSystem",
    "QuasiGeostrophicSiteElement",
    "StommelBoxesElement",
    "StommelSystem",
]
