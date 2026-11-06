"""Celestial mechanics: orbits of bodies under gravity, geodesics around masses, and charges in magnetic fields."""

from ._dipoles import DipoleChargeElement, stormer_problem
from ._geodesic_particle import GeodesicParticleElement
from ._geodesics import kerr_frame_dragging, light_bending, schwarzschild_precession
from ._kepler_body import KeplerBodyElement
from ._orbits import n_body, restricted_three_body, sitnikov, two_body_orbit
from ._rotating_frame_body import RotatingFrameBodyElement
from ._sitnikov_body import SitnikovBodyElement
from ._spacetimes import KerrSpace, SchwarzschildSpace

__all__: list[str] = [
    'DipoleChargeElement',
    'GeodesicParticleElement',
    'KeplerBodyElement',
    'KerrSpace',
    'RotatingFrameBodyElement',
    'SchwarzschildSpace',
    'SitnikovBodyElement',
    'kerr_frame_dragging',
    'light_bending',
    'n_body',
    'restricted_three_body',
    'schwarzschild_precession',
    'sitnikov',
    'stormer_problem',
    'two_body_orbit',
]
