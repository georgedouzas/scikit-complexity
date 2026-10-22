"""Collective behaviour models: active walkers that form trails by changing the ground they walk on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, cast

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element, Field
from ....core.laws import Law, Rate, Rule, Symbols, laplacian
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import System

GROUND = 'land/ground.G'
TRAIL = 'land/ground.V'


def _corners(width: int, height: int) -> list[tuple[int, int]]:
    """Return the four corner cells of a grid, the entry points of the walkers."""
    return [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]


class Ground(Element):
    """Ground whose structure `G` is raised by footprints and recovers, with a trail potential `V` that smooths it.

    The structure relaxes back to bare ground as `G' = -G / T`, and walkers add footprints. The trail potential
    follows `V' = D ΔV + (G - V) / tau`, so it settles to the structure smoothed over a length `sqrt(D tau)`, the
    distance from which walkers notice a trail.

    Args:
        label:
            The label of the ground or the lattice field.

        G:
            The initial ground structure.

        V:
            The initial trail potential.

        T:
            The durability of trails, the time over which they fade.

        D:
            The diffusion of the trail potential.

        tau:
            The relaxation time of the trail potential.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import Ground
        >>> Ground('ground').T
        200.0
    """

    G: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    V: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    T: float = parameter(default=200.0, domain=Domain.POSITIVE)
    D: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the fading of the structure and the smoothing of the trail potential.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.G, -s.G / s.T), Rate(s.V, s.D * laplacian(s.V) + (s.G - s.V) / s.tau)]


class Walker(Element):
    """Pedestrian who walks toward a goal, drawn to existing trails, and leaves footprints that build them.

    The walker heads along the unit vector toward its goal plus `attraction` times the gradient of the trail potential
    `V`, after Helbing, Keltsch, and Molnár. Every step it moves to a neighbouring cell with a probability that grows
    with `exp(directedness * cos angle)`, where the angle is between the step and the heading, and adds the footprint
    `intensity (1 - G / G_max)` to the ground. At the goal it picks another entry point at random.

    Args:
        label:
            The label of the agent or group.

        goal_x:
            The column of the goal.

        goal_y:
            The row of the goal.

        directedness:
            How strongly the walker heads for its goal.

        attraction:
            How strongly trails draw the walker.

        intensity:
            The footprint on bare ground.

        G_max:
            The largest ground structure.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import Walker
        >>> Walker.group('walkers', 4, location=(0, 0)).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'goal_x'})
    goal_x: float = state(default=0.0)
    goal_y: float = state(default=0.0)
    directedness: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    attraction: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    intensity: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    G_max: float = parameter(default=5.0, domain=Domain.POSITIVE)
    width: float = parameter(default=30.0, domain=Domain.POSITIVE)
    height: float = parameter(default=30.0, domain=Domain.POSITIVE)

    def _choose_goal(self, agent: AgentView, model: ModelView) -> None:
        """Pick another entry point as the new goal."""
        here = cast('tuple[int, int]', agent.location)
        goals = [corner for corner in _corners(round(agent.width), round(agent.height)) if corner != here]
        agent.goal_x, agent.goal_y = (float(value) for value in goals[int(model.rng.integers(len(goals)))])

    def _step_weights(self, agent: AgentView, model: ModelView, cells: list[tuple[int, int]]) -> np.ndarray:
        """Return the unnormalised probabilities of stepping to each neighbouring cell."""
        x, y = cast('tuple[int, int]', agent.location)
        goal = np.array([agent.goal_x - x, agent.goal_y - y])
        steps = np.array([(cell[0] - x, cell[1] - y) for cell in cells], dtype=float)
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        rises = np.array([model.field(TRAIL, cell) for cell in cells]) - model.field(TRAIL, (x, y))
        slope = 2 * (steps / lengths[:, None] ** 2).T @ rises / len(cells)
        heading = goal / max(float(np.hypot(*goal)), 1e-12) + agent.attraction * slope
        alignment = steps @ heading / (lengths * max(float(np.hypot(*heading)), 1e-12))
        return np.exp(agent.directedness * (alignment - alignment.max()))

    def laws(self, s: Symbols) -> list[Law]:
        """Return the walking rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.walk)]

    def walk(self, agent: AgentView, model: ModelView) -> None:
        """Step toward the goal along trails, leave a footprint, and pick a new goal on arrival.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if agent.location == (round(agent.goal_x), round(agent.goal_y)):
            self._choose_goal(agent, model)
            return
        cells = cast('list[tuple[int, int]]', list(agent.neighbor_cells()))
        weights = self._step_weights(agent, model, cells)
        cell = cells[model.rng.choice(len(cells), p=weights / weights.sum())]
        agent.move_to(cell)
        model.deposit(GROUND, cell, agent.intensity * (1 - model.field(GROUND, cell) / agent.G_max))


@register_model(
    field='biology/collective_behaviour',
    tags=('agents', 'field', 'trail formation', 'co-simulation'),
    references=('Helbing, Keltsch, and Molnár 1997', 'arXiv:cond-mat/0012229'),
    defaults={'until': 300, 'dt': 1},
)
def active_walkers(
    size: int = 30,
    *,
    n_walkers: int = 20,
    attraction: float = 1.0,
    T: float = 200.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active walkers who walk between the corners of a green and wear a system of trails into it.

    Walkers prefer existing trails to the direct way, and trails fade unless they are used, so a few shared trails
    emerge that bundle the ways between the corners, a compromise between short and comfortable routes.

    Args:
        size:
            The number of rows and columns of the green.

        n_walkers:
            The number of walkers.

        attraction:
            How strongly trails draw the walkers.

        T:
            The durability of trails.

        random_state:
            The seed or generator of the starting corners and goals.

    Returns:
        system:
            The model, with the walkers `people/walkers`, the ground structure `land/ground.G`, and the trail potential
            `land/ground.V`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import active_walkers
        >>> [child.label for child in active_walkers().systems]
        ['people', 'land']
    """
    rng = np.random.default_rng(random_state)
    corners = _corners(size, size)
    starts = rng.integers(len(corners), size=n_walkers)
    goals = [corners[(start + 1 + int(rng.integers(len(corners) - 1))) % len(corners)] for start in starts]
    walkers = Walker.group(
        'walkers',
        n_walkers,
        location=[corners[start] for start in starts],
        goal_x=np.array([float(goal[0]) for goal in goals]),
        goal_y=np.array([float(goal[1]) for goal in goals]),
        attraction=attraction,
        width=float(size),
        height=float(size),
    )
    ground = Field('ground', Ground, LatticeSpace((size, size), boundary='reflecting'), values={'T': T})
    return System(
        components=[
            System('people', components=[walkers], space=LatticeSpace((size, size), boundary='fixed')),
            System('land', components=[ground]),
        ],
    )
