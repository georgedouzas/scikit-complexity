"""Relativity shared by the physics packages.

Spacetimes derive their geometry from a metric with EinsteinPy, and free bodies follow their geodesics.
"""

from ._geodesic import GeodesicInteraction
from ._spacetime import SpacetimeSpace

__all__: list[str] = [
    'GeodesicInteraction',
    'SpacetimeSpace',
]
