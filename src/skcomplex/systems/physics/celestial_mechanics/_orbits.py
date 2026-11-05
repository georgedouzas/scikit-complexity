"""Celestial mechanics systems: orbits of bodies under gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, sqrt

from ....core.components.interactions import GradientInteraction, HamiltonianInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver, Yoshida4Solver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ..common.particles import GravityInteraction, MassPointElement
from ._kepler_body import KeplerBodyElement
from ._rotating_frame_body import RotatingFrameBodyElement
from ._sitnikov_body import SitnikovBodyElement
from .types import Preset

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


def _get_kepler_energy(body: SimpleNamespace) -> Expression:
    """Return the energy `|p|^2 / 2 - G M / |q|` per unit mass of a planet relative to its star."""
    return (body.px**2 + body.py**2) / 2 - body.G * body.mass / sqrt(body.x**2 + body.y**2)


def _get_kepler_hamiltonian(s: SimpleNamespace) -> Expression:
    """Return the Hamiltonian of the Kepler problem of the target."""
    return _get_kepler_energy(s.target)


def _get_rotating_frame_energy(body: SimpleNamespace) -> Expression:
    """Return the Hamiltonian of a light body in the frame that rotates with two primaries at unit angular velocity."""
    first = sqrt((body.x + body.mu) ** 2 + body.y**2)
    second = sqrt((body.x - 1 + body.mu) ** 2 + body.y**2)
    kinetic = (body.px**2 + body.py**2) / 2 + body.y * body.px - body.x * body.py
    return kinetic - (1 - body.mu) / first - body.mu / second


def _get_rotating_frame_hamiltonian(s: SimpleNamespace) -> Expression:
    """Return the Hamiltonian of the rotating frame of the target."""
    return _get_rotating_frame_energy(s.target)


def _get_kepler_observables(system: System) -> list[ObservableSpec]:
    """Return the energy of the planet."""
    del system
    return [ObservableSpec('energy', _get_kepler_energy, element='planet')]


def _get_rotating_frame_observables(system: System) -> list[ObservableSpec]:
    """Return the Hamiltonian of the light body, minus half its Jacobi constant."""
    del system
    return [ObservableSpec('energy', _get_rotating_frame_energy, element='body')]


def _get_sitnikov_potential(s: SimpleNamespace) -> Expression:
    """Return the potential `-1 / sqrt(z² + r(t)²)` of the two primaries at the distance `r(t)` from the axis."""
    distance = (1 - s.target.e * cos(s.t)) / 2
    return -1 / sqrt(s.target.z**2 + distance**2)


def _preset(name: str) -> Preset:
    """Return the masses, positions, velocities, and gravitational constant of a named configuration."""
    if name not in PRESETS:
        error_msg = f'Unknown preset {name!r}. Use one of {", ".join(sorted(PRESETS))}.'
        raise ValidationError(error_msg)
    return PRESETS[name]


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'gravity', 'hamiltonian'),
    references=('Kepler 1609', 'Newton 1687'),
    defaults={'until': 6.0, 'dt': 0.01, 'solver': Yoshida4Solver(), 'invariants': {'planet.energy': 1e-6}},
    observables=_get_kepler_observables,
)
def two_body_orbit(
    eccentricity: float = 0.0,
    *,
    G: float = 1.0,
    mass: float = 1000.0,
    radius: float = 10.0,
) -> System:
    """Return a light planet orbiting a heavy star under Newtonian gravity, as the Kepler problem of their separation.

    The planet moves in the coordinate `(x, y)` relative to the star with the canonical momenta `(px, py)` per unit
    mass, under the Hamiltonian `H = (px² + py²) / 2 - G M / sqrt(x² + y²)`, which is separable. The energy and the
    angular momentum `x py - y px` are conserved, and a bound orbit is an ellipse with the star at a focus.

    Args:
        eccentricity:
            The eccentricity of the orbit, between 0 for a circle and 1 for an escape.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

        radius:
            The distance of the planet from the star at its closest approach.

    Returns:
        system:
            The system, with the coordinates `planet.x` and `planet.y` and the momenta `planet.px` and `planet.py`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import two_body_orbit
        >>> [element.label for element in two_body_orbit().elements]
        ['planet']
    """
    speed = float(np.sqrt(G * mass * (1 + eccentricity) / radius))
    return System(
        elements=[KeplerBodyElement('planet', x=radius, y=0.0, px=0.0, py=speed, G=G, mass=mass)],
        interactions=[
            HamiltonianInteraction(
                'gravity',
                target='planet',
                coordinates=('x', 'y'),
                momenta=('px', 'py'),
                hamiltonian=_get_kepler_hamiltonian,
            ),
        ],
    )


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'gravity', 'chaos'),
    references=('Argun et al. 2021', 'Dilão 2023', 'Chenciner and Montgomery 2000'),
    defaults={'until': 6.33, 'dt': 0.01},
)
def n_body(
    preset: str = 'figure_eight',
    *,
    masses: Sequence[float] | None = None,
    positions: Sequence[Sequence[float]] | None = None,
    velocities: Sequence[Sequence[float]] | None = None,
    G: float | None = None,
) -> System:
    """Return bodies in the plane that attract each other pairwise by Newtonian gravity.

    The preset `'figure_eight'` is the periodic orbit of three equal masses found by Chenciner and Montgomery, with
    period 6.3259 for `G = 1`. The preset `'inner_solar_system'` is the Sun with Mercury, Venus, the Earth, and Mars on
    circular orbits, in astronomical units and years, with `G = 4 π²`. Masses, positions, velocities, or `G` given
    explicitly replace those of the preset.

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

    Returns:
        system:
            The system, with the particles `body_<i>` in two dimensions.

    Raises:
        ValidationError:
            The preset is unknown.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import n_body
        >>> len(n_body('inner_solar_system').interactions)
        10
    """
    preset_G, preset_masses, preset_positions, preset_velocities = _preset(preset)
    masses = preset_masses if masses is None else masses
    positions = preset_positions if positions is None else positions
    velocities = preset_velocities if velocities is None else velocities
    G = preset_G if G is None else G
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
    return System(elements=bodies, interactions=attractions, space=ContinuousSpace(n_dim=2))


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'gravity', 'chaos', 'rotating frame', 'hamiltonian'),
    references=('Dilão 2023', 'Szebehely 1967'),
    defaults={'until': 20.0, 'dt': 0.02, 'solver': RK4Solver(), 'invariants': {'body.energy': 1e-6}},
    observables=_get_rotating_frame_observables,
)
def restricted_three_body(
    x: float = 0.5,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 1.15,
    *,
    mu: float = 0.01215,
) -> System:
    """Return a light body moving in the plane of two primaries on circular orbits, in the frame that rotates with them.

    The primaries of masses `1 - μ` and `μ` sit at `(-μ, 0)` and `(1 - μ, 0)`, and the light body moves under the
    Hamiltonian `H = (px² + py²) / 2 + y px - x py - (1 - μ) / r1 - μ / r2` of the rotating frame, with the momenta
    `px = x' - y` and `py = y' + x`. Its terms `y px - x py` couple the coordinates and the momenta, so the Hamiltonian
    is not separable. It equals minus half the Jacobi constant `x² + y² + 2 (1 - μ) / r1 + 2 μ / r2 - x'² - y'²`, which
    is conserved. The default `μ` is that of the Earth and the Moon.

    The default start has the Jacobi constant 2.835, below that of the neck towards the Moon, and its orbit is chaotic:
    it passes near the Moon after 3.5 time units and wanders between the two primaries, with a Lyapunov exponent near
    0.2. Starts of a larger constant are confined around the Earth, such as `vy = 0.8` with the constant 3.52, which
    traces a regular precessing rosette.

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

    Returns:
        system:
            The system, with the coordinates `body.x` and `body.y` and the momenta `body.px` and `body.py`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import restricted_three_body
        >>> restricted_three_body(mu=0.1).elements[0].mu
        0.1
    """
    return System(
        elements=[RotatingFrameBodyElement('body', x=x, y=y, px=vx - y, py=vy + x, mu=mu)],
        interactions=[
            HamiltonianInteraction(
                'motion',
                target='body',
                coordinates=('x', 'y'),
                momenta=('px', 'py'),
                hamiltonian=_get_rotating_frame_hamiltonian,
            ),
        ],
    )


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'gravity', 'chaos'),
    references=('Dilão 2023', 'Sitnikov 1960'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def sitnikov(z: float = 0.5, vz: float = 0.0, *, e: float = 0.0) -> System:
    """Return a light body moving on the axis through the centre of mass of two equal primaries, normal to their orbit.

    The primaries of mass 1/2 orbit their centre of mass with unit mean separation and period `2π`, at distance
    `r(t) = (1 - e cos t) / 2` from it to first order in the eccentricity `e`, and the light body follows
    `z'' = -z / (z² + r(t)²)^(3/2)`. For circular primaries the energy `z'²/2 - 1 / √(z² + 1/4)` is conserved and the
    motion is integrable, and eccentric primaries make bounded orbits chaotic.

    Args:
        z:
            The initial height above the plane of the primaries.

        vz:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

    Returns:
        system:
            The system, with the height `body.z` of order 2.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import sitnikov
        >>> sitnikov(e=0.1).elements[0].e
        0.1
    """
    return System(
        elements=[SitnikovBodyElement('body', z=z, z_d1=vz, e=e)],
        interactions=[GradientInteraction('gravity', target='body', states='z', potential=_get_sitnikov_potential)],
    )
