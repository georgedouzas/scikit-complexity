"""Geodesic systems: test particles and light rays falling freely around a static or a spinning mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import QQ, SR, Expression, cos, matrix, sin
from sage.structure.element import Matrix

from ....core.components.elements import Element
from ....core.components.interactions import GeodesicInteraction
from ....core.components.spaces import PseudoRiemannianSpace
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ValidationError
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system

R_INDEX = 1
PHI_INDEX = 3


def _equatorial_position(x: npt.NDArray[np.float64], **_: float) -> list[list[float]]:
    """Return the position `(r cos phi, r sin phi)` of a particle in the equatorial plane."""
    return [[x[R_INDEX] * np.cos(x[PHI_INDEX]), x[R_INDEX] * np.sin(x[PHI_INDEX])]]


def _central_mass(**_: float) -> list[list[float]]:
    """Return the position of the central mass at the origin."""
    return [[0.0, 0.0]]


class GeodesicParticleElement(Element):
    """Test particle at the spacetime position `x = (t, r, theta, phi)` around a central mass, with its four-velocity.

    The position is a vector state of order 2 in the Schwarzschild or Boyer-Lindquist coordinates of the space, and its
    companion `x_d1` is the four-velocity, the derivative of the position by the proper time of a massive particle or by
    the affine parameter of a light ray. The particle is drawn in the equatorial plane, at `(r cos phi, r sin phi)`.

    Args:
        label:
            The label of the particle.

        x:
            The initial position `(t, r, theta, phi)`.

        x_d1:
            The initial four-velocity.

        visual:
            How the particle is drawn, or `None` for its class visual.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import GeodesicParticleElement
        >>> GeodesicParticleElement('particle', x=[0.0, 10.0, 1.5708, 0.0]).x
        [0.0, 10.0, 1.5708, 0.0]
    """

    visual = VisualSpec('bodies', {'positions': _equatorial_position, 'trail': None, 'marks': _central_mass})

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2)] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1


class SchwarzschildSpace(PseudoRiemannianSpace):
    """Spacetime outside a static, spherical mass `M`, the Schwarzschild metric in units with `G = c = 1`.

    The line element in the coordinates `(t, r, theta, phi)` is
    `ds² = -(1 - 2M/r) dt² + dr² / (1 - 2M/r) + r² (dθ² + sin²θ dφ²)`, with the horizon at `r = 2M`. The mass enters
    the metric as an exact rational, which keeps the symbolic derivation fast.

    Args:
        mass:
            The mass `M`.

        curvature:
            Whether `prepare` derives the curvature tensors besides the Christoffel symbols.

        label:
            The label of the space in its system.

    Raises:
        ValidationError:
            The mass is not positive.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SchwarzschildSpace
        >>> SchwarzschildSpace(mass=1.0, curvature=False).prepare().christoffel_[1][0][0]
        (r - 2)/r^3
    """

    def __init__(self, mass: float = 1.0, *, curvature: bool = False, label: str = 'space') -> None:
        super().__init__(curvature=curvature, label=label)
        self.mass = mass

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in ('t', 'r', 'theta', 'phi'))

    def _get_metric(self) -> Matrix:
        """Return the Schwarzschild metric of the mass.

        Returns:
            metric:
                The metric in the coordinates `(t, r, theta, phi)`.

        Raises:
            ValidationError:
                The mass is not positive.
        """
        if not self.mass > 0:
            error_msg = f'The mass of {type(self).__name__} is {self.mass!r}. Give a positive mass.'
            raise ValidationError(error_msg)
        _, r, theta, _ = self._get_coordinates()
        lapse = 1 - 2 * QQ(self.mass) / r
        return matrix(SR, 4, 4, {(0, 0): -lapse, (1, 1): 1 / lapse, (2, 2): r**2, (3, 3): r**2 * sin(theta) ** 2})


class KerrSpace(PseudoRiemannianSpace):
    """Spacetime outside a spinning mass `M` of angular momentum `M a`, the Kerr metric in Boyer-Lindquist coordinates.

    With `Σ = r² + a² cos²θ` and `Δ = r² - 2 M r + a²`, the line element in `(t, r, theta, phi)` is
    `ds² = -(1 - 2Mr/Σ) dt² - (4 M a r sin²θ / Σ) dt dφ + Σ/Δ dr² + Σ dθ² + (r² + a² + 2 M a² r sin²θ / Σ) sin²θ dφ²`,
    in units with `G = c = 1`. The term in `dt dφ` drags the frames of free particles around the spin axis. The mass and
    the spin enter the metric as exact rationals, which keeps the symbolic derivation fast.

    Args:
        mass:
            The mass `M`.

        spin:
            The spin parameter `a`, the angular momentum per unit mass, between `-M` and `M`.

        curvature:
            Whether `prepare` derives the curvature tensors besides the Christoffel symbols, which takes long.

        label:
            The label of the space in its system.

    Raises:
        ValidationError:
            The mass is not positive, or the spin exceeds the mass.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import KerrSpace
        >>> KerrSpace(mass=1.0, spin=0.5).prepare().metric_[0, 3]
        -4*r*sin(theta)^2/(4*r^2 + cos(theta)^2)
    """

    def __init__(
        self,
        mass: float = 1.0,
        spin: float = 0.5,
        *,
        curvature: bool = False,
        label: str = 'space',
    ) -> None:
        super().__init__(curvature=curvature, label=label)
        self.mass = mass
        self.spin = spin

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in ('t', 'r', 'theta', 'phi'))

    def _get_metric(self) -> Matrix:
        """Return the Kerr metric of the mass and the spin.

        Returns:
            metric:
                The metric in the coordinates `(t, r, theta, phi)`.

        Raises:
            ValidationError:
                The mass is not positive, or the spin exceeds the mass.
        """
        if not self.mass > 0 or not abs(self.spin) <= self.mass:
            error_msg = (
                f'{type(self).__name__} has the mass {self.mass!r} and the spin {self.spin!r}. Give a positive mass '
                'and a spin no larger than it.'
            )
            raise ValidationError(error_msg)
        _, r, theta, _ = self._get_coordinates()
        mass, spin = QQ(self.mass), QQ(self.spin)
        sigma = r**2 + spin**2 * cos(theta) ** 2
        delta = r**2 - 2 * mass * r + spin**2
        drag = -2 * mass * spin * r * sin(theta) ** 2 / sigma
        return matrix(
            SR,
            4,
            4,
            {
                (0, 0): -(1 - 2 * mass * r / sigma),
                (0, 3): drag,
                (3, 0): drag,
                (1, 1): sigma / delta,
                (2, 2): sigma,
                (3, 3): (r**2 + spin**2 + 2 * mass * spin**2 * r * sin(theta) ** 2 / sigma) * sin(theta) ** 2,
            },
        )


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
