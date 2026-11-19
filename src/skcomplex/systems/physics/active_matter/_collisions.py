"""Flocking by binary collisions.

Self-propelled particles align their headings when they meet.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet
from scipy.spatial import KDTree

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class FlockerView(AgentView):
    """Flocker as rules see it, with its heading.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    x: float
    y: float
    theta: float


class FlockerElement(Element):
    """Self-propelled point particles at the points `x` and `y` of a periodic square, with the headings `theta`.

    The heading wraps around the circle.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial first coordinate, one per particle or shared.

        y:
            The initial second coordinate, one per particle or shared.

        theta:
            The initial heading, one per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import FlockerElement
        >>> FlockerElement('bird', theta=1.0).theta
        1.0
    """

    visual = VisualSpec('agents', {'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)})

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        y: object = 0.0,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.y = y
        self.theta = theta

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return FlockerView


class BinaryCollisionsInteraction(Interaction):
    """Motion, self-diffusion, and aligning collisions of self-propelled particles.

    The collisions are binary, and the particles move in steps of `1 / substeps`.

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
        >>> from skcomplex.systems.physics.active_matter import BinaryCollisionsInteraction
        >>> BinaryCollisionsInteraction('collisions', target='birds').d0
        0.5
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        lam: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        d0: Annotated[float, RealSet.open(0, oo)] = 0.5,
        substeps: Annotated[float, RealSet.open(0, oo)] = 10.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.lam = lam
        self.sigma = sigma
        self.d0 = d0
        self.substeps = substeps

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the whole system that moves the particles for one unit of time.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context, self.fly)]

    def fly(self, system: SystemView) -> None:
        """Move the particles for one unit of time.

        Every substep has self-diffusion and collisions.

        Args:
            system:
                The system.
        """
        agents = system.get_agents(self.target, FlockerView)
        size = np.asarray(system.dimensions, dtype=float)
        positions = np.array([(agent.x, agent.y) for agent in agents], dtype=np.float64)
        headings = np.array([agent.theta for agent in agents], dtype=np.float64)
        speed, lam, sigma, d0 = (system.get_parameter(name) for name in ('speed', 'lam', 'sigma', 'd0'))
        substeps = max(1, round(system.get_parameter('substeps')))
        tau, rng = 1.0 / substeps, system.rng
        touching = KDTree(positions, boxsize=size).query_pairs(d0)
        for _ in range(substeps):
            positions = np.mod(positions + speed * tau * np.column_stack([np.cos(headings), np.sin(headings)]), size)
            kicked = rng.random(len(headings)) < -math.expm1(-lam * tau)
            headings[kicked] += rng.normal(0.0, sigma, int(kicked.sum()))
            now = KDTree(positions, boxsize=size).query_pairs(d0)
            BinaryCollisionsInteraction._collide(headings, sorted(now - touching), sigma, rng)
            touching = now
        for agent, position, heading in zip(agents, positions, np.mod(headings, 2 * math.pi), strict=True):
            agent.x, agent.y = float(position[0]), float(position[1])
            agent.theta = float(heading)

    @staticmethod
    def _collide(headings: Array, pairs: list[tuple[int, int]], sigma: float, rng: np.random.Generator) -> None:
        """Set both headings of every pair to their mean plus independent Gaussian noise.

        The pairs come in random order.
        """
        for index in rng.permutation(len(pairs)):
            first, second = pairs[index]
            mean = math.atan2(
                math.sin(headings[first]) + math.sin(headings[second]),
                math.cos(headings[first]) + math.cos(headings[second]),
            )
            headings[first], headings[second] = mean + rng.normal(0.0, sigma, 2)


def _get_cosine(agent: FlockerView) -> float:
    """Return the first component of an agent's heading."""
    return cos(agent.theta)


def _get_sine(agent: FlockerView) -> float:
    """Return the second component of an agent's heading."""
    return sin(agent.theta)


def _get_polarization(s: SimpleNamespace) -> Expression:
    """Return the length of the mean heading, the polarization."""
    return (s.mean_cos**2 + s.mean_sin**2) ** 0.5


class BinaryCollisionFlockingSystem(BaseSystem):
    """Self-propelled particles that flock through aligning binary collisions.

    This is the Bertin-Droz-Grégoire system.

    Particles fly straight at a constant speed. Their headings jump by Gaussian angles in self-diffusion events at the
    rate `lam`, and two particles that come closer than `d0` both take their mean heading plus Gaussian noise. The
    hydrodynamic equations derived from the Boltzmann equation predict collective motion above the density `rho_t = pi
    lam (1 - exp(-sigma^2 / 2)) / (4 (exp(-sigma^2 / 2) - 2/3))`, for `speed = 1` and `d0 = 1/2`, so collective motion
    needs `sigma < sqrt(2 ln(3/2))`, about 0.90, at any density. The system has the agent group `birds` and the
    observables `mean_cos`, `mean_sin`, and `polarization`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import BinaryCollisionFlockingSystem
        >>> BinaryCollisionFlockingSystem(10).prepare().system_.interactions[0].sigma
        0.3
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'agents', 'flocking', 'collisions', 'boltzmann equation')
    references = ('Bertin, Droz, and Grégoire 2006', 'Bertin, Droz, and Grégoire 2009', 'Bertin 2021')
    until = 200
    dt = 1
    observables = (
        Observable('mean_cos', _get_cosine, group='birds'),
        Observable('mean_sin', _get_sine, group='birds'),
        Observable('polarization', _get_polarization),
    )

    def __init__(
        self,
        n_particles: int = 400,
        *,
        size: float = 20.0,
        speed: float = 1.0,
        lam: float = 1.0,
        sigma: float = 0.3,
        d0: float = 0.5,
        substeps: int = 10,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.speed = speed
        self.lam = lam
        self.sigma = sigma
        self.d0 = d0
        self.substeps = substeps
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of self-propelled particles that flock through aligning binary collisions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        headings = rng.uniform(0.0, 2 * math.pi, self.n_particles)
        positions = rng.uniform(0.0, self.size, (self.n_particles, 2))
        birds = FlockerElement(
            'birds',
            x=positions[:, 0],
            y=positions[:, 1],
            theta=headings,
            size=self.n_particles,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        collisions = BinaryCollisionsInteraction(
            'collisions',
            target='birds',
            speed=self.speed,
            lam=self.lam,
            sigma=self.sigma,
            d0=self.d0,
            substeps=float(self.substeps),
        )
        self.system_ = System(self.label, elements=[birds], interactions=[collisions])
