"""Motion, self-diffusion, and aligning binary collisions of self-propelled particles, in steps of `1 / substeps`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.spatial import KDTree

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array
from ._flocker_view import FlockerView


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
        >>> from skcomplex.systems.physics.active_matter import BinaryCollisionsInteraction
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

    def fly(self, system: SystemView) -> None:
        """Move the particles for one unit of time, with self-diffusion and collisions in every substep.

        Args:
            system:
                The system.
        """
        agents = system.agents(self.target, FlockerView)
        size = np.asarray(system.dimensions, dtype=float)
        positions = np.array([agent.location for agent in agents], dtype=float)
        headings = np.array([agent.theta for agent in agents], dtype=float)
        speed, lam, sigma, d0 = (system.parameter(name) for name in ('speed', 'lam', 'sigma', 'd0'))
        substeps = max(1, round(system.parameter('substeps')))
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
            agent.move_to(position)
            agent.theta = float(heading)

    @staticmethod
    def _collide(headings: Array, pairs: list[tuple[int, int]], sigma: float, rng: np.random.Generator) -> None:
        """Set both headings of every pair, in random order, to their mean heading plus independent Gaussian noise."""
        for index in rng.permutation(len(pairs)):
            first, second = pairs[index]
            mean = math.atan2(
                math.sin(headings[first]) + math.sin(headings[second]),
                math.cos(headings[first]) + math.cos(headings[second]),
            )
            headings[first], headings[second] = mean + rng.normal(0.0, sigma, 2)
