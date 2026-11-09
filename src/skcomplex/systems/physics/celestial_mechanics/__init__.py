"""Celestial mechanics systems.

The package covers orbits under gravity and relativity, spin-orbit coupling, and charges in magnetic fields.
"""

from ._dipoles import DipoleChargeElement, StormerProblemSystem
from ._geodesics import (
    GeodesicParticleElement,
    KerrFrameDraggingSystem,
    KerrSpace,
    LightBendingSystem,
    SchwarzschildPrecessionSystem,
    SchwarzschildSpace,
)
from ._orbits import (
    KeplerBodyElement,
    NBodySystem,
    RestrictedThreeBodySystem,
    RotatingFrameBodyElement,
    SitnikovBodyElement,
    SitnikovSystem,
    TwoBodyOrbitSystem,
)
from ._spin_orbit import DumbbellElement, KeplerianDumbbellSystem, RelativisticOrbitSystem

__all__: list[str] = [
    'DipoleChargeElement',
    'DumbbellElement',
    'GeodesicParticleElement',
    'KeplerBodyElement',
    'KeplerianDumbbellSystem',
    'KerrFrameDraggingSystem',
    'KerrSpace',
    'LightBendingSystem',
    'NBodySystem',
    'RelativisticOrbitSystem',
    'RestrictedThreeBodySystem',
    'RotatingFrameBodyElement',
    'SchwarzschildPrecessionSystem',
    'SchwarzschildSpace',
    'SitnikovBodyElement',
    'SitnikovSystem',
    'StormerProblemSystem',
    'TwoBodyOrbitSystem',
]
