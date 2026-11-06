"""Geodesic systems: test particles and light rays falling freely around a static or a spinning mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ....core.components.interactions import GeodesicInteraction
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._geodesic_particle import GeodesicParticleElement
from ._spacetimes import KerrSpace, SchwarzschildSpace

EQUATOR = math.pi / 2
"""The polar angle of the equatorial plane, where the orbits stay."""

GEODESIC_SOLVER = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
"""The adaptive solver that keeps the conserved energy, angular momentum, and norm of a geodesic to high accuracy."""

@register_system(
    field='physics/celestial_mechanics',
    tags=('general relativity', 'ode', 'second order', 'gravity', 'geodesics'),
    references=('Schwichtenberg 2020', 'Satz 2022', 'Schwarzschild 1916', 'Einstein 1915', 'Carroll 2004'),
    defaults={'until': 20000.0, 'dt': 10.0, 'solver': GEODESIC_SOLVER},
)
def schwarzschild_precession(
    semi_major_axis: float = 100.0,
    eccentricity: float = 0.5,
    *,
    mass: float = 1.0,
) -> System:
    """Return a massive particle on a bound orbit around a static mass, whose perihelion advances every turn.

    The particle follows a timelike geodesic of the Schwarzschild spacetime `SchwarzschildSpace`, in units with
    `G = c = 1` and with its proper time as the time of the system. It starts at the periapsis
    `r₁ = a (1 - e)` of an orbit whose apoapsis is `r₂ = a (1 + e)`, with the energy `E` and the angular momentum `L`
    for which both are turning points of `E² = (1 - 2M/r) (1 + L²/r²) + (dr/dτ)²`. Unlike a Kepler ellipse, the orbit
    does not close: the periapsis advances by `6 π M / (a (1 - e²))` per turn when the orbit is far from the
    horizon, or `6 π G M / (c² a (1 - e²))` in other units, which accounts for the anomalous 43 arcseconds per century
    of Mercury.

    Args:
        semi_major_axis:
            The semi-major axis `a` of the orbit.

        eccentricity:
            The eccentricity `e` of the orbit, in `[0, 1)`.

        mass:
            The mass `M` of the central body.

    Returns:
        system:
            The system, with the position `particle.x = (t, r, theta, phi)` and the four-velocity `particle.x_d1`.

    Raises:
        ValidationError:
            The eccentricity is outside `[0, 1)`, or the orbit comes closer than the innermost stable circular orbit.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import schwarzschild_precession
        >>> round(schwarzschild_precession().elements[0].x[1], 6)
        50.0
    """
    if not 0 <= eccentricity < 1:
        error_msg = f'The eccentricity is {eccentricity!r}. Give a value in [0, 1) for a bound orbit.'
        raise ValidationError(error_msg)
    periapsis, apoapsis = semi_major_axis * (1 - eccentricity), semi_major_axis * (1 + eccentricity)
    if periapsis <= 6 * mass:
        error_msg = (
            f'The periapsis {periapsis} lies inside the innermost stable circular orbit at 6 M = {6 * mass}. Give a '
            'larger semi-major axis or a smaller eccentricity.'
        )
        raise ValidationError(error_msg)
    lapses = [1 - 2 * mass / radius for radius in (periapsis, apoapsis)]
    if eccentricity == 0:
        momentum_squared = mass * periapsis**2 / (periapsis - 3 * mass)
    else:
        momentum_squared = (lapses[1] - lapses[0]) / (lapses[0] / periapsis**2 - lapses[1] / apoapsis**2)
    energy = math.sqrt(lapses[0] * (1 + momentum_squared / periapsis**2))
    particle = GeodesicParticleElement(
        'particle',
        x=[0.0, periapsis, EQUATOR, 0.0],
        x_d1=[energy / lapses[0], 0.0, 0.0, math.sqrt(momentum_squared) / periapsis**2],
    )
    space = SchwarzschildSpace(mass=mass)
    geodesic = GeodesicInteraction('geodesic', target='particle', space=space)
    return System(elements=[particle], interactions=[geodesic], space=space)


@register_system(
    field='physics/celestial_mechanics',
    tags=('general relativity', 'ode', 'second order', 'gravity', 'geodesics', 'light'),
    references=('Schwichtenberg 2020', 'Satz 2022', 'Einstein 1915', 'Dyson, Eddington, and Davidson 1920'),
    defaults={'until': 4000.0, 'dt': 4.0, 'solver': GEODESIC_SOLVER},
)
def light_bending(
    impact_parameter: float = 20.0,
    distance: float = 2000.0,
    *,
    mass: float = 1.0,
) -> System:
    """Return a light ray that passes a static mass and is deflected towards it.

    The ray follows a null geodesic of the Schwarzschild spacetime `SchwarzschildSpace`, in units with `G = c = 1`, with
    the affine parameter as the time of the system. It starts at the distance `D` along the axis behind the mass,
    offset by the impact parameter `b`, aimed parallel to the axis, with the energy `E = 1` and the angular momentum
    `L = b`. Far from the mass the ray bends by `4 M / b`, twice the Newtonian deflection of a particle at the speed of
    light, as the eclipse of 1919 measured for starlight that grazes the Sun.

    Args:
        impact_parameter:
            The impact parameter `b`, the distance of the unbent ray from the mass.

        distance:
            The distance `D` along the axis from which the ray starts.

        mass:
            The mass `M` of the central body.

    Returns:
        system:
            The system, with the position `photon.x = (t, r, theta, phi)` and the tangent `photon.x_d1`.

    Raises:
        ValidationError:
            The ray would be captured by the mass, its impact parameter being at most `3 √3 M`, or it starts closer
            than its impact parameter.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import light_bending
        >>> round(light_bending().elements[0].x_d1[0], 4)
        1.001
    """
    if impact_parameter <= 3 * math.sqrt(3) * mass:
        error_msg = (
            f'The impact parameter {impact_parameter} is at most 3 √3 M = {3 * math.sqrt(3) * mass}, so the mass '
            'captures the ray. Give a larger impact parameter.'
        )
        raise ValidationError(error_msg)
    if distance <= impact_parameter:
        error_msg = f'The distance {distance} is not larger than the impact parameter {impact_parameter}.'
        raise ValidationError(error_msg)
    radius = math.hypot(distance, impact_parameter)
    lapse = 1 - 2 * mass / radius
    radial = -math.sqrt(1 - lapse * impact_parameter**2 / radius**2)
    photon = GeodesicParticleElement(
        'photon',
        x=[0.0, radius, EQUATOR, math.atan2(impact_parameter, -distance)],
        x_d1=[1 / lapse, radial, 0.0, -impact_parameter / radius**2],
    )
    space = SchwarzschildSpace(mass=mass)
    geodesic = GeodesicInteraction('geodesic', target='photon', space=space)
    return System(elements=[photon], interactions=[geodesic], space=space)


@register_system(
    field='physics/celestial_mechanics',
    tags=('general relativity', 'ode', 'second order', 'gravity', 'geodesics', 'frame dragging'),
    references=('Schwichtenberg 2020', 'Kerr 1963', 'Bardeen, Press, and Teukolsky 1972'),
    defaults={'until': 33.0, 'dt': 0.1, 'solver': GEODESIC_SOLVER},
)
def kerr_frame_dragging(
    radius: float = 10.0,
    *,
    mass: float = 1.0,
    spin: float = 0.9,
) -> System:
    """Return a particle without angular momentum that falls into a spinning mass and is dragged around it.

    The particle follows a timelike geodesic of the Kerr spacetime `KerrSpace` in the equatorial plane, in units with
    `G = c = 1`, with its proper time as the time of the system. It starts at rest in the radius at `r₀`, with zero
    angular momentum `L = g_tφ dt/dτ + g_φφ dφ/dτ = 0`. A particle without angular momentum around a static mass falls
    straight in, but the spin drags the frames around it: the particle turns with the angular velocity
    `dφ/dt = -g_tφ / g_φφ = 2 M a / (r³ + a² r + 2 M a²)` of the frames at its radius, faster as it falls.

    Args:
        radius:
            The initial radius `r₀`.

        mass:
            The mass `M` of the central body.

        spin:
            The spin parameter `a`, between `-M` and `M`.

    Returns:
        system:
            The system, with the position `particle.x = (t, r, theta, phi)` and the four-velocity `particle.x_d1`.

    Raises:
        ValidationError:
            The particle starts inside the ergosphere, at a radius of at most `2 M`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import kerr_frame_dragging
        >>> round(kerr_frame_dragging().elements[0].x_d1[3], 5)
        0.00199
    """
    if radius <= 2 * mass:
        error_msg = f'The radius {radius} is inside the ergosphere at 2 M = {2 * mass}. Give a larger radius.'
        raise ValidationError(error_msg)
    space = KerrSpace(mass=mass, spin=spin)
    g_tt = -(1 - 2 * mass / radius)
    g_tphi = -2 * mass * spin / radius
    g_phiphi = radius**2 + spin**2 + 2 * mass * spin**2 / radius
    omega = -g_tphi / g_phiphi
    rate = 1 / math.sqrt(-(g_tt + 2 * g_tphi * omega + g_phiphi * omega**2))
    particle = GeodesicParticleElement(
        'particle',
        x=[0.0, radius, EQUATOR, 0.0],
        x_d1=[rate, 0.0, 0.0, omega * rate],
    )
    geodesic = GeodesicInteraction('geodesic', target='particle', space=space)
    return System(elements=[particle], interactions=[geodesic], space=space)
