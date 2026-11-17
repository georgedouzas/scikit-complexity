"""Celestial mechanics systems of bodies orbiting under gravity.

`FIGURE_EIGHT` holds the gravitational constant, masses, positions, and velocities of the figure-eight orbit.
`INNER_SOLAR_SYSTEM` holds the gravitational constant, masses, positions, and velocities of the Sun and the inner
planets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.systems import BaseSystem as OldBaseSystem
from ....core.components.systems import CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver, Yoshida4Solver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, Role, StateSpec, VisualSpec
from ....core.systems import BaseSystem, System
from ...physics.common.mechanics import HamiltonianInteraction
from .types import Preset

FIGURE_EIGHT: Preset = (
    1.0,
    [1.0, 1.0, 1.0],
    [[-0.97000436, 0.24308753], [0.97000436, -0.24308753], [0.0, 0.0]],
    [[0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146]],
)
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
PRESETS = {'figure_eight': FIGURE_EIGHT, 'inner_solar_system': INNER_SOLAR_SYSTEM}


def _get_star(**_: float) -> list[list[float]]:
    """Return the position of the star at the origin of the relative coordinate."""
    return [[0.0, 0.0]]


class KeplerBodyElement(Element):
    """Planet at `(x, y)` relative to a heavy star, with canonical momenta `(px, py)`.

    The planet is that of the Kepler problem.

    The star of mass `mass` sits at the origin.

    Args:
        label:
            The label of the planet.

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

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _get_star})

    def __init__(
        self,
        label: str,
        *,
        x: float = 10.0,
        y: float = 0.0,
        px: float = 0.0,
        py: float = 10.0,
        G: Annotated[float, RealSet.open(0, oo)] = 1.0,
        mass: Annotated[float, RealSet.open(0, oo)] = 1000.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.G = G
        self.mass = mass


def _get_primaries(mu: float, **_: float) -> list[list[float]]:
    """Return the positions of the two primaries in the rotating frame."""
    return [[-mu, 0.0], [1 - mu, 0.0]]


class RotatingFrameBodyElement(Element):
    """Light body at `(x, y)` with canonical momenta `(px, py)` in a rotating frame.

    The frame rotates with two primaries.

    The states are of order 1, and the momenta are `px = x' - y` and `py = y' + x` for the unit angular velocity of the
    frame. The primaries of masses `1 - mu` and `mu` sit at `(-mu, 0)` and `(1 - mu, 0)`.

    Args:
        label:
            The label of the body.

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

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _get_primaries})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 0.5,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01215,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.mu = mu


class SitnikovBodyElement(Element):
    """Light body at the height `z` on the axis of two equal primaries.

    The axis runs through the centre of mass of the primaries, which orbit with the eccentricity `e`. The vertical
    velocity is `z'`.

    Args:
        label:
            The label of the body.

        z:
            The initial height.

        e:
            The eccentricity of the primaries' orbits.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SitnikovBodyElement
        >>> SitnikovBodyElement('body', e=0.1).e
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        z: float = 0.5,
        e: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.z = z
        self.e = e


class SitnikovGravityInteraction(Interaction):
    """Pull of two equal primaries on a light body on their axis, `z'' = -dV/dz`.

    The primaries of mass 1/2 sit at the distance `r(t) = (1 - e cos t) / 2` from the axis, with the eccentricity `e` of
    the target, so the body at the height `z` has the potential `V = -1 / sqrt(z² + r(t)²)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the body.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SitnikovGravityInteraction
        >>> SitnikovGravityInteraction('gravity', target='body').target
        'body'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the height of the body.

        Args:
            context:
                The namespace of the body and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            body, t = context.target, context.t
            height = SR.symbol('_z')
            distance = (1 - body.e * cos(t)) / 2
            potential = -1 / sqrt(height**2 + distance**2)
            self.laws_ = [RateLaw(body.z.diff(t, 2), -potential.diff(height).subs({height: body.z}))]


class GravityInteraction(Interaction):
    """Newtonian gravity between two particles.

    The energy is `-G * m_first * m_second / |x_first - x_second|`, and its gradient moves both particles: the first
    follows `x_first'' = -G m_second (x_first - x_second) / |x_first - x_second|³` and the second the opposite pull
    `x_second'' = G m_first (x_first - x_second) / |x_first - x_second|³`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        G:
            The gravitational constant.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics._orbits import GravityInteraction
        >>> GravityInteraction('gravity', first='sun', second='earth', G=1.0).G
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        G: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.G = G

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the accelerations of the positions `x` of both particles, whose masses are `m`.

        Args:
            context:
                The namespace of the particles, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            first, second = context.first, context.second
            gap = first.x - second.x
            pull = context.G * gap / sqrt(gap.dot_product(gap)) ** 3
            self.laws_ = [RateLaw(first.x, -second.m * pull, order=2), RateLaw(second.x, first.m * pull, order=2)]


def _get_kepler_energy(body: SimpleNamespace) -> Expression:
    """Return the energy `|p|^2 / 2 - G M / |q|` per unit mass of a planet.

    The planet moves relative to its star.
    """
    return (body.px**2 + body.py**2) / 2 - body.G * body.mass / sqrt(body.x**2 + body.y**2)


class TwoBodyOrbitSystem(BaseSystem):
    """A light planet orbiting a heavy star under Newtonian gravity.

    The system is the Kepler problem of their separation.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    invariants: ClassVar[Mapping[str, float]] = {'planet.energy': 1e-6}
    observables = (Observable('energy', _get_kepler_energy, element='planet'),)

    def __init__(
        self,
        eccentricity: float = 0.0,
        *,
        G: float = 1.0,
        mass: float = 1000.0,
        radius: float = 10.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.eccentricity = eccentricity
        self.G = G
        self.mass = mass
        self.radius = radius

    @staticmethod
    def _get_kepler_hamiltonian(s: SimpleNamespace) -> Expression:
        """Return the Hamiltonian of the Kepler problem of the target."""
        return _get_kepler_energy(s.target)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a light planet orbiting a heavy star, integrated by the symplectic Yoshida method.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        speed = float(np.sqrt(self.G * self.mass * (1 + self.eccentricity) / self.radius))
        self.system_ = System(
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
            solver=Yoshida4Solver(),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.preset = preset
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.G = G

    @staticmethod
    def _get_preset(name: str) -> Preset:
        """Return a preset's masses, positions, velocities, and gravitational constant.

        The preset is a named configuration.
        """
        if name not in PRESETS:
            error_msg = f'Unknown preset {name!r}. Use one of {", ".join(sorted(PRESETS))}.'
            raise ValidationError(error_msg)
        return PRESETS[name]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of bodies in the plane that attract pairwise by Newtonian gravity.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        preset_G, preset_masses, preset_positions, preset_velocities = self._get_preset(self.preset)
        masses = preset_masses if self.masses is None else self.masses
        positions = preset_positions if self.positions is None else self.positions
        velocities = preset_velocities if self.velocities is None else self.velocities
        G = preset_G if self.G is None else self.G
        bodies = [
            Element(f'body_{index}', values={'x': list(position), "x'": list(velocity), 'm': mass})
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
        self.system_ = System(self.label, elements=bodies, interactions=attractions, space=ContinuousSpace(n_dim=2))


def _get_rotating_frame_energy(body: SimpleNamespace) -> Expression:
    """Return the Hamiltonian of a light body in the frame rotating with two primaries.

    The frame rotates at unit angular velocity.
    """
    first = sqrt((body.x + body.mu) ** 2 + body.y**2)
    second = sqrt((body.x - 1 + body.mu) ** 2 + body.y**2)
    kinetic = (body.px**2 + body.py**2) / 2 + body.y * body.px - body.x * body.py
    return kinetic - (1 - body.mu) / first - body.mu / second


class RestrictedThreeBodySystem(OldBaseSystem):
    """A light body moving in the plane of two primaries on circular orbits.

    The body moves in the frame that rotates with the primaries.

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
    invariants: ClassVar[Mapping[str, float]] = {'body.energy': 1e-6}
    observables = (Observable('energy', _get_rotating_frame_energy, element='body'),)

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

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        return CompositeSystem(
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
    """A light body moving on the axis through the mass centre of two equal primaries.

    The axis is normal to their orbit.

    The primaries of mass 1/2 orbit their centre of mass with unit mean separation and period `2π`, at distance `r(t) =
    (1 - e cos t) / 2` from it to first order in the eccentricity `e`, and the light body follows `z'' = -z / (z² +
    r(t)²)^(3/2)`. For circular primaries the energy `z'²/2 - 1 / √(z² + 1/4)` is conserved and the motion is
    integrable, and eccentric primaries make bounded orbits chaotic. The system has the height `body.z` and the vertical
    velocity `body.z'`.

    Args:
        z:
            The initial height above the plane of the primaries.

        vz:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.z = z
        self.vz = vz
        self.e = e

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a light body on the axis of two equal primaries.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[SitnikovBodyElement('body', z=self.z, e=self.e, values={"z'": self.vz})],
            interactions=[SitnikovGravityInteraction('gravity', target='body')],
        )
