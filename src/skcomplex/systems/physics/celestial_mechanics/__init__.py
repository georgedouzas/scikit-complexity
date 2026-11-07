"""Celestial mechanics: orbits under gravity and relativity, spin-orbit coupling, and charges in magnetic fields."""

from ._dipoles import DipoleChargeElement, stormer_problem
from ._geodesics import (
    GeodesicParticleElement,
    KerrSpace,
    SchwarzschildSpace,
    kerr_frame_dragging,
    light_bending,
    schwarzschild_precession,
)
from ._orbits import (
    KeplerBodyElement,
    RotatingFrameBodyElement,
    SitnikovBodyElement,
    n_body,
    restricted_three_body,
    sitnikov,
    two_body_orbit,
)
from ._spin_orbit import DumbbellElement, RelativisticOrbitElement, keplerian_dumbbell, relativistic_orbit

__all__: list[str] = [
    'DipoleChargeElement',
    'DumbbellElement',
    'GeodesicParticleElement',
    'KeplerBodyElement',
    'KerrSpace',
    'RelativisticOrbitElement',
    'RotatingFrameBodyElement',
    'SchwarzschildSpace',
    'SitnikovBodyElement',
    'keplerian_dumbbell',
    'kerr_frame_dragging',
    'light_bending',
    'n_body',
    'relativistic_orbit',
    'restricted_three_body',
    'schwarzschild_precession',
    'sitnikov',
    'stormer_problem',
    'two_body_orbit',
]
