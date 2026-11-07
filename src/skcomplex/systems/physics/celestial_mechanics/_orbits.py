"""Celestial mechanics systems: orbits of bodies under gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import GradientInteraction, HamiltonianInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver, Yoshida4Solver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.particles import MassPointElement, PotentialEnergyInteraction
from .types import Preset


def _star(**_: float) -> list[list[float]]:
    """Return the position of the star at the origin of the relative coordinate."""
    return [[0.0, 0.0]]


class KeplerBodyElement(Element):
    """Planet at `(x, y)` relative to a heavy star, with the canonical momenta `(px, py)` of the Kepler problem.

    The states are of order 1, and the star of mass `mass` sits at the origin.

    Args:
        label:
            The label of the planet.

        visual:
            How the planet is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum per unit mass.

        py:
            The initial second momentum per unit mass.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import KeplerBodyElement
        >>> KeplerBodyElement('planet', mass=10.0).mass
        10.0
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _star})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 10.0,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 10.0,
        G: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mass: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.G = G
        self.mass = mass


def _primaries(mu: float, **_: float) -> list[list[float]]:
    """Return the positions of the two primaries in the rotating frame."""
    return [[-mu, 0.0], [1 - mu, 0.0]]


class RotatingFrameBodyElement(Element):
    """Light body at `(x, y)` with the canonical momenta `(px, py)` in the frame that rotates with two primaries.

    The states are of order 1, and the momenta are `px = x' - y` and `py = y' + x` for the unit angular velocity of the
    frame. The primaries of masses `1 - mu` and `mu` sit at `(-mu, 0)` and `(1 - mu, 0)`.

    Args:
        label:
            The label of the body.

        visual:
            How the body is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

        mu:
            The mass of the lighter primary, as a fraction of the total.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RotatingFrameBodyElement
        >>> RotatingFrameBodyElement('body', mu=0.1).mu
        0.1
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _primaries})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 0.5,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01215,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.mu = mu


class SitnikovBodyElement(Element):
    """Light body at the height `z`, of order 2, on the axis through the centre of mass of two equal primaries.

    The primaries orbit with the eccentricity `e`.

    Args:
        label:
            The label of the body.

        visual:
            How the body is drawn, or `None` for none.

        z:
            The initial height.

        z_d1:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SitnikovBodyElement
        >>> SitnikovBodyElement('body', e=0.1).e
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(order=2)] = 0.5,
        z_d1: float = 0.0,
        e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.z_d1 = z_d1
        self.e = e


FIGURE_EIGHT: Preset = (
    1.0,
    [1.0, 1.0, 1.0],
    [[-0.97000436, 0.24308753], [0.97000436, -0.24308753], [0.0, 0.0]],
    [[0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146]],
)
"""Gravitational constant, masses, positions, and velocities of the figure-eight orbit."""

INNER_SOLAR_SYSTEM: Preset = (
    4 * math.pi**2,
    [1.0, 1.66e-7, 2.45e-6, 3.0e-6, 3.23e-7],
    [[0.0, 0.0], [0.387, 0.0], [0.723, 0.0], [1.0, 0.0], [1.524, 0.0]],
    [
        [0.0, 0.0],
        [0.0, 2 * math.pi / math.sqrt(0.387)],
        [0.0, 2 * math.pi / math.sqrt(0.723)],
        [0.0, 2 * math.pi],
        [0.0, 2 * math.pi / math.sqrt(1.524)],
    ],
)
"""Gravitational constant, masses, positions, and velocities of the Sun and the inner planets."""

PRESETS = {'figure_eight': FIGURE_EIGHT, 'inner_solar_system': INNER_SOLAR_SYSTEM}


class GravityInteraction(PotentialEnergyInteraction):
    """Newtonian gravity between two particles.

    The energy is `-G * m_first * m_second / |x_first - x_second|`, and its gradient moves both particles.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        G:
            The gravitational constant.

        states:
            The name of the vector state that holds the positions of the particles.

        divisor:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics._orbits import GravityInteraction
        >>> GravityInteraction('gravity', first='sun', second='earth', G=1.0).G
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        G: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        states: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=first, states=states, divisor=divisor)
        self.first = first
        self.second = second
        self.G = G

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `-G * m_first * m_second / |x_first - x_second|`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        gap = getattr(s.first, self._get_names()[0]) - getattr(s.second, self._get_names()[0])
        masses = getattr(s.first, str(self.divisor)) * getattr(s.second, str(self.divisor))
        return -s.G * masses / sqrt(gap.dot_product(gap))


def _get_kepler_energy(body: SimpleNamespace) -> Expression:
    """Return the energy `|p|^2 / 2 - G M / |q|` per unit mass of a planet relative to its star."""
    return (body.px**2 + body.py**2) / 2 - body.G * body.mass / sqrt(body.x**2 + body.y**2)


def _get_rotating_frame_energy(body: SimpleNamespace) -> Expression:
    """Return the Hamiltonian of a light body in the frame that rotates with two primaries at unit angular velocity."""
    first = sqrt((body.x + body.mu) ** 2 + body.y**2)
    second = sqrt((body.x - 1 + body.mu) ** 2 + body.y**2)
    kinetic = (body.px**2 + body.py**2) / 2 + body.y * body.px - body.x * body.py
    return kinetic - (1 - body.mu) / first - body.mu / second


class TwoBodyOrbitSystem(BaseSystem):
    """A light planet orbiting a heavy star under Newtonian gravity, as the Kepler problem of their separation.

    The planet moves in the coordinate `(x, y)` relative to the star with the canonical momenta `(px, py)` per unit
    mass, under the Hamiltonian `H = (px² + py²) / 2 - G M / sqrt(x² + y²)`, which is separable. The energy and the
    angular momentum `x py - y px` are conserved, and a bound orbit is an ellipse with the star at a focus. The system
    has the coordinates `planet.x` and `planet.y` and the momenta `planet.px` and `planet.py`.

    Args:
        eccentricity:
            The eccentricity of the orbit, between 0 for a circle and 1 for an escape.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

        radius:
            The distance of the planet from the star at its closest approach.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import TwoBodyOrbitSystem
        >>> [element.label for element in TwoBodyOrbitSystem().prepare().system_.elements]
        ['planet']
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'gravity', 'hamiltonian')
    references = ('Kepler 1609', 'Newton 1687')
    until = 6.0
    dt = 0.01
    solver = Yoshida4Solver()
    observables = (ObservableSpec('energy', _get_kepler_energy, element='planet'),)

    def __init__(
        self,
        eccentricity: float = 0.0,
        *,
        G: float = 1.0,
        mass: float = 1000.0,
        radius: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.eccentricity = eccentricity
        self.G = G
        self.mass = mass
        self.radius = radius

    @staticmethod
    def _get_kepler_hamiltonian(s: SimpleNamespace) -> Expression:
        """Return the Hamiltonian of the Kepler problem of the target."""
        return _get_kepler_energy(s.target)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        speed = float(np.sqrt(self.G * self.mass * (1 + self.eccentricity) / self.radius))
        return System(
            self.label,
            elements=[KeplerBodyElement('planet', x=self.radius, y=0.0, px=0.0, py=speed, G=self.G, mass=self.mass)],
            interactions=[
                HamiltonianInteraction(
                    'gravity',
                    target='planet',
                    coordinates=('x', 'y'),
                    momenta=('px', 'py'),
                    hamiltonian=self._get_kepler_hamiltonian,
                ),
            ],
        )


class NBodySystem(BaseSystem):
    """Bodies in the plane that attract each other pairwise by Newtonian gravity.

    The preset `'figure_eight'` is the periodic orbit of three equal masses found by Chenciner and Montgomery, with
    period 6.3259 for `G = 1`. The preset `'inner_solar_system'` is the Sun with Mercury, Venus, the Earth, and Mars on
    circular orbits, in astronomical units and years, with `G = 4 π²`. Masses, positions, velocities, or `G` given
    explicitly replace those of the preset. The system has the particles `body_<i>` in two dimensions.

    Args:
        preset:
            `'figure_eight'` or `'inner_solar_system'`.

        masses:
            The masses of the bodies, or `None` for those of the preset.

        positions:
            The initial positions of the bodies, or `None` for those of the preset.

        velocities:
            The initial velocities of the bodies, or `None` for those of the preset.

        G:
            The gravitational constant, or `None` for that of the preset.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The preset is unknown.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import NBodySystem
        >>> len(NBodySystem('inner_solar_system').prepare().system_.interactions)
        10
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'gravity', 'chaos')
    references = ('Argun et al. 2021', 'Dilão 2023', 'Chenciner and Montgomery 2000')
    until = 6.33
    dt = 0.01

    def __init__(
        self,
        preset: str = 'figure_eight',
        *,
        masses: Sequence[float] | None = None,
        positions: Sequence[Sequence[float]] | None = None,
        velocities: Sequence[Sequence[float]] | None = None,
        G: float | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.preset = preset
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.G = G

    @staticmethod
    def _preset(name: str) -> Preset:
        """Return the masses, positions, velocities, and gravitational constant of a named configuration."""
        if name not in PRESETS:
            error_msg = f'Unknown preset {name!r}. Use one of {", ".join(sorted(PRESETS))}.'
            raise ValidationError(error_msg)
        return PRESETS[name]

    def _build_system(self) -> System:
        """Return bodies in the plane that attract each other pairwise by Newtonian gravity."""
        preset_G, preset_masses, preset_positions, preset_velocities = self._preset(self.preset)
        masses = preset_masses if self.masses is None else self.masses
        positions = preset_positions if self.positions is None else self.positions
        velocities = preset_velocities if self.velocities is None else self.velocities
        G = preset_G if self.G is None else self.G
        bodies = [
            MassPointElement(f'body_{index}', x=list(position), x_d1=list(velocity), m=mass)
            for index, (mass, position, velocity) in enumerate(zip(masses, positions, velocities, strict=True))
        ]
        attractions = [
            GravityInteraction(
                f'gravity_{first}_{second}',
                first=f'body_{first}',
                second=f'body_{second}',
                G=G,
            )
            for first, second in itertools.combinations(range(len(bodies)), 2)
        ]
        return System(self.label, elements=bodies, interactions=attractions, space=ContinuousSpace(n_dim=2))


class RestrictedThreeBodySystem(BaseSystem):
    """A light body moving in the plane of two primaries on circular orbits, in the frame that rotates with them.

    The primaries of masses `1 - μ` and `μ` sit at `(-μ, 0)` and `(1 - μ, 0)`, and the light body moves under the
    Hamiltonian `H = (px² + py²) / 2 + y px - x py - (1 - μ) / r1 - μ / r2` of the rotating frame, with the momenta
    `px = x' - y` and `py = y' + x`. Its terms `y px - x py` couple the coordinates and the momenta, so the Hamiltonian
    is not separable. It equals minus half the Jacobi constant `x² + y² + 2 (1 - μ) / r1 + 2 μ / r2 - x'² - y'²`, which
    is conserved. The default `μ` is that of the Earth and the Moon.

    The default start has the Jacobi constant 2.835, below that of the neck towards the Moon, and its orbit is chaotic:
    it passes near the Moon after 3.5 time units and wanders between the two primaries, with a Lyapunov exponent near
    0.2. Starts of a larger constant are confined around the Earth, such as `vy = 0.8` with the constant 3.52, which
    traces a regular precessing rosette. The system has the coordinates `body.x` and `body.y` and the momenta `body.px`
    and `body.py`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        mu:
            The mass of the lighter primary, as a fraction of the total.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RestrictedThreeBodySystem
        >>> RestrictedThreeBodySystem(mu=0.1).prepare().system_.elements[0].mu
        0.1
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'gravity', 'chaos', 'rotating frame', 'hamiltonian')
    references = ('Dilão 2023', 'Szebehely 1967')
    until = 20.0
    dt = 0.02
    solver = RK4Solver()
    observables = (ObservableSpec('energy', _get_rotating_frame_energy, element='body'),)

    def __init__(
        self,
        x: float = 0.5,
        y: float = 0.0,
        vx: float = 0.0,
        vy: float = 1.15,
        *,
        mu: float = 0.01215,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mu = mu

    @staticmethod
    def _get_rotating_frame_hamiltonian(s: SimpleNamespace) -> Expression:
        """Return the Hamiltonian of the rotating frame of the target."""
        return _get_rotating_frame_energy(s.target)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                RotatingFrameBodyElement(
                    'body',
                    x=self.x,
                    y=self.y,
                    px=self.vx - self.y,
                    py=self.vy + self.x,
                    mu=self.mu,
                ),
            ],
            interactions=[
                HamiltonianInteraction(
                    'motion',
                    target='body',
                    coordinates=('x', 'y'),
                    momenta=('px', 'py'),
                    hamiltonian=self._get_rotating_frame_hamiltonian,
                ),
            ],
        )


class SitnikovSystem(BaseSystem):
    """A light body moving on the axis through the centre of mass of two equal primaries, normal to their orbit.

    The primaries of mass 1/2 orbit their centre of mass with unit mean separation and period `2π`, at distance `r(t) =
    (1 - e cos t) / 2` from it to first order in the eccentricity `e`, and the light body follows `z'' = -z / (z² +
    r(t)²)^(3/2)`. For circular primaries the energy `z'²/2 - 1 / √(z² + 1/4)` is conserved and the motion is
    integrable, and eccentric primaries make bounded orbits chaotic. The system has the height `body.z` of order 2.

    Args:
        z:
            The initial height above the plane of the primaries.

        vz:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SitnikovSystem
        >>> SitnikovSystem(e=0.1).prepare().system_.elements[0].e
        0.1
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'second order', 'gravity', 'chaos')
    references = ('Dilão 2023', 'Sitnikov 1960')
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        z: float = 0.5,
        vz: float = 0.0,
        *,
        e: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.z = z
        self.vz = vz
        self.e = e

    @staticmethod
    def _get_sitnikov_potential(s: SimpleNamespace) -> Expression:
        """Return the potential `-1 / sqrt(z² + r(t)²)` of the two primaries at the distance `r(t)` from the axis."""
        distance = (1 - s.target.e * cos(s.t)) / 2
        return -1 / sqrt(s.target.z**2 + distance**2)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[SitnikovBodyElement('body', z=self.z, z_d1=self.vz, e=self.e)],
            interactions=[
                GradientInteraction('gravity', target='body', states='z', potential=self._get_sitnikov_potential),
            ],
        )
