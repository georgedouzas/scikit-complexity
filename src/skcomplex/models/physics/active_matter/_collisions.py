"""Flocking by binary collisions: self-propelled particles that align their headings when they meet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet
from scipy.spatial import KDTree

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array


def _cosine(agent: AgentView) -> float:
    """Return the first component of an agent's heading."""
    return math.cos(agent.theta)


def _sine(agent: AgentView) -> float:
    """Return the second component of an agent's heading."""
    return math.sin(agent.theta)


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the length of the mean heading, the polarization."""
    return (s.mean_cos**2 + s.mean_sin**2) ** 0.5


def _collide(headings: Array, pairs: list[tuple[int, int]], sigma: float, rng: np.random.Generator) -> None:
    """Set both headings of every pair, in random order, to their mean heading plus independent Gaussian noise."""
    for index in rng.permutation(len(pairs)):
        first, second = pairs[index]
        mean = math.atan2(
            math.sin(headings[first]) + math.sin(headings[second]),
            math.cos(headings[first]) + math.cos(headings[second]),
        )
        headings[first], headings[second] = mean + rng.normal(0.0, sigma, 2)


class FlockerElement(Element):
    """Self-propelled point particle with a heading `theta`, in a periodic square.

    Args:
        label:
            The label of the agent or group.

        theta:
            The initial heading.

    Examples:
        >>> from skcomplex.models.physics.active_matter import FlockerElement
        >>> FlockerElement('bird', theta=1.0).theta
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)}),
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta


class BinaryCollisionsInteraction(Interaction):
    """Motion, self-diffusion, and aligning binary collisions of self-propelled particles, in steps of `1 / substeps`.

    In every substep of length `tau` each particle moves `speed * tau` along its heading, and with probability
    `1 - exp(-lam * tau)` its heading jumps by a Gaussian angle of standard deviation `sigma`. Two particles collide
    when their distance falls below `d0` during the substep, and both headings become the mean heading
    `arg(exp(i theta_1) + exp(i theta_2))` plus independent Gaussian angles of standard deviation `sigma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of particles.

        speed:
            The self-propulsion speed.

        lam:
            The rate of self-diffusion events.

        sigma:
            The standard deviation of the angular noise of both kinds of events.

        d0:
            The collision distance.

        substeps:
            The number of substeps per unit of time.

    Examples:
        >>> from skcomplex.models.physics.active_matter import BinaryCollisionsInteraction
        >>> BinaryCollisionsInteraction('collisions', target='birds').d0
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        d0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        substeps: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.lam = lam
        self.sigma = sigma
        self.d0 = d0
        self.substeps = substeps

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that moves the particles for one unit of time.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.fly)]

    def fly(self, model: ModelView) -> None:
        """Move the particles for one unit of time, with self-diffusion and collisions in every substep.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        size = np.asarray(model.dimensions, dtype=float)
        positions = np.array([agent.location for agent in agents], dtype=float)
        headings = np.array([agent.theta for agent in agents], dtype=float)
        speed, lam, sigma, d0 = (model.parameter(name) for name in ('speed', 'lam', 'sigma', 'd0'))
        substeps = max(1, round(model.parameter('substeps')))
        tau, rng = 1.0 / substeps, model.rng
        touching = KDTree(positions, boxsize=size).query_pairs(d0)
        for _ in range(substeps):
            positions = np.mod(positions + speed * tau * np.column_stack([np.cos(headings), np.sin(headings)]), size)
            kicked = rng.random(len(headings)) < -math.expm1(-lam * tau)
            headings[kicked] += rng.normal(0.0, sigma, int(kicked.sum()))
            now = KDTree(positions, boxsize=size).query_pairs(d0)
            _collide(headings, sorted(now - touching), sigma, rng)
            touching = now
        for agent, position, heading in zip(agents, positions, np.mod(headings, 2 * math.pi), strict=True):
            agent.move_to(position)
            agent.theta = float(heading)


def _flocking_observables(system: System) -> list[ObservableSpec]:
    """Return the mean heading components and the polarization of the birds."""
    return [
        ObservableSpec('mean_cos', _cosine, group='birds'),
        ObservableSpec('mean_sin', _sine, group='birds'),
        ObservableSpec('polarization', _polarization),
    ]


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'agents', 'flocking', 'collisions', 'boltzmann equation'),
    references=('Bertin, Droz, and Grégoire 2006', 'Bertin, Droz, and Grégoire 2009', 'Bertin 2021'),
    defaults={'until': 200, 'dt': 1},
    observables=_flocking_observables,
)
def binary_collision_flocking(
    n_particles: int = 400,
    *,
    size: float = 20.0,
    speed: float = 1.0,
    lam: float = 1.0,
    sigma: float = 0.3,
    d0: float = 0.5,
    substeps: int = 10,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return self-propelled particles that flock through aligning binary collisions, the Bertin-Droz-Grégoire model.

    Particles fly straight at a constant speed. Their headings jump by Gaussian angles in self-diffusion events at the
    rate `lam`, and two particles that come closer than `d0` both take their mean heading plus Gaussian noise. The
    hydrodynamic equations derived from the Boltzmann equation predict collective motion above the density
    `rho_t = pi lam (1 - exp(-sigma^2 / 2)) / (4 (exp(-sigma^2 / 2) - 2/3))`, for `speed = 1` and `d0 = 1/2`, so
    collective motion needs `sigma < sqrt(2 ln(3/2))`, about 0.90, at any density.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic square.

        speed:
            The self-propulsion speed.

        lam:
            The rate of self-diffusion events.

        sigma:
            The standard deviation of the angular noise.

        d0:
            The collision distance.

        substeps:
            The number of substeps per unit of time.

        random_state:
            The seed or generator of the uniform initial positions and headings.

    Returns:
        system:
            The model, with the agent group `birds` and the observables `mean_cos`, `mean_sin`, and `polarization`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import binary_collision_flocking
        >>> binary_collision_flocking(10).interactions[0].sigma
        0.3
    """
    rng = np.random.default_rng(random_state)
    birds = GroupElement(
        'birds',
        element=FlockerElement,
        size=n_particles,
        values={'theta': rng.uniform(0.0, 2 * math.pi, n_particles)},
        location=rng.uniform(0.0, size, (n_particles, 2)),
    )
    collisions = BinaryCollisionsInteraction(
        'collisions',
        target='birds',
        speed=speed,
        lam=lam,
        sigma=sigma,
        d0=d0,
        substeps=float(substeps),
    )
    return System(
        elements=[birds],
        interactions=[collisions],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
