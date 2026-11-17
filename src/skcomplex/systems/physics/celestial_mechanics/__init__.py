"""Celestial mechanics systems.

The package covers orbits under gravity and relativity, spin-orbit coupling, and charges in magnetic fields.
"""

from ._dipoles import DipoleChargeElement, DipoleLorentzInteraction, StormerProblemSystem
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
    RotatingFrameMotionInteraction,
    SitnikovBodyElement,
    SitnikovGravityInteraction,
    SitnikovSystem,
    TwoBodyOrbitSystem,
)
from ._spin_orbit import (
    DumbbellElement,
    DumbbellMotionInteraction,
    EinsteinOrbitInteraction,
    KeplerianDumbbellSystem,
    RelativisticOrbitSystem,
)

__all__: list[str] = [
    'DipoleChargeElement',
    'DipoleLorentzInteraction',
    'DumbbellElement',
    'DumbbellMotionInteraction',
    'EinsteinOrbitInteraction',
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
    'RotatingFrameMotionInteraction',
    'SchwarzschildPrecessionSystem',
    'SchwarzschildSpace',
    'SitnikovBodyElement',
    'SitnikovGravityInteraction',
    'SitnikovSystem',
    'StormerProblemSystem',
    'TwoBodyOrbitSystem',
]
