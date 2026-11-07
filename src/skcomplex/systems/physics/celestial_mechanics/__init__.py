"""Celestial mechanics: orbits under gravity and relativity, spin-orbit coupling, and charges in magnetic fields."""

from ._dipoles import DipoleChargeElement, stormer_problem
from ._dumbbell import DumbbellElement
from ._geodesic_particle import GeodesicParticleElement
from ._geodesics import kerr_frame_dragging, light_bending, schwarzschild_precession
from ._kepler_body import KeplerBodyElement
from ._orbits import n_body, restricted_three_body, sitnikov, two_body_orbit
from ._relativistic_orbit import RelativisticOrbitElement
from ._rotating_frame_body import RotatingFrameBodyElement
from ._sitnikov_body import SitnikovBodyElement
from ._spacetimes import KerrSpace, SchwarzschildSpace
from ._spin_orbit import keplerian_dumbbell, relativistic_orbit

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
