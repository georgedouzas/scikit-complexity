"""Collective behaviour models: active walkers that form trails by changing the ground they walk on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import FieldElement, GroupElement
from ....core.interactions import (
    DiffusionInteraction,
    LifeCycleInteraction,
    MapInteraction,
    MovementInteraction,
    RelaxationInteraction,
    TransferInteraction,
)
from ....core.laws import Source
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ..._catalog import register_model
from ._ground import GroundElement
from ._walker import WalkerElement
from ._walker_view import WalkerView

TRAIL = 'land/ground.V'


def _corners(width: int, height: int) -> list[tuple[int, int]]:
    """Return the four corner cells of a grid, the entry points of the walkers."""
    return [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]


def _get_durability(namespace: SimpleNamespace) -> Expression:
    """Return the durability `T` of trails, the time over which the structure fades to bare ground."""
    durability: Expression = namespace.target.T
    return durability


def _get_diffusion(namespace: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `D` of the trail potential."""
    coefficient: Expression = namespace.target.D
    return coefficient


def _get_structure(namespace: SimpleNamespace) -> Expression:
    """Return the ground structure `G`, the goal of the trail potential."""
    structure: Expression = namespace.target.G
    return structure


def _get_relaxation(namespace: SimpleNamespace) -> Expression:
    """Return the relaxation time `tau` of the trail potential."""
    time: Expression = namespace.target.tau
    return time


def _get_goal(walker: AgentViewProtocol | None, model: ModelViewProtocol) -> dict[str, object]:
    """Return another entry point than the cell of a walker as its goal, drawn at random."""
    if not isinstance(walker, WalkerView):
        return {}
    here = cast('tuple[int, int]', walker.location)
    goals = [corner for corner in _corners(round(walker.width), round(walker.height)) if corner != here]
    goal_x, goal_y = goals[int(model.rng.integers(len(goals)))]
    return {'goal_x': float(goal_x), 'goal_y': float(goal_y)}


def _arrived(walker: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether a walker stands at its goal."""
    del model
    return isinstance(walker, WalkerView) and walker.location == (round(walker.goal_x), round(walker.goal_y))


def _waits(walker: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether a walker arrived at its goal in the previous step, so it leaves the green now."""
    return _arrived(walker, model) and isinstance(walker, WalkerView) and bool(walker.arrived)


def _walks(walker: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether a walker is away from its goal, so it steps."""
    return not _arrived(walker, model)


def _get_arrived(walker: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return whether a walker stands at its goal at the end of its activation."""
    return float(_arrived(walker, model))


def _get_step(walker: AgentViewProtocol, model: ModelViewProtocol) -> object:
    """Return a neighbouring cell drawn with a weight that grows with the alignment of the step with the heading.

    The heading is the unit vector toward the goal plus `attraction` times the gradient of the trail potential, and
    the weight of a step is `exp(directedness * cos angle)`, where the angle is between the step and the heading.
    """
    if not isinstance(walker, WalkerView):
        return None
    x, y = cast('tuple[int, int]', walker.location)
    cells = cast('list[tuple[int, int]]', list(walker.neighbor_cells()))
    goal = np.array([walker.goal_x - x, walker.goal_y - y])
    steps = np.array([(cell[0] - x, cell[1] - y) for cell in cells], dtype=float)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    rises = np.array([model.field(TRAIL, cell) for cell in cells]) - model.field(TRAIL, (x, y))
    slope = 2 * (steps / lengths[:, None] ** 2).T @ rises / len(cells)
    heading = goal / max(float(np.hypot(*goal)), 1e-12) + walker.attraction * slope
    alignment = steps @ heading / (lengths * max(float(np.hypot(*heading)), 1e-12))
    weights = np.exp(walker.directedness * (alignment - alignment.max()))
    return cells[int(model.rng.choice(len(cells), p=weights / weights.sum()))]


def _steps(walker: AgentViewProtocol, cell: object, model: ModelViewProtocol) -> bool:
    """Return whether a walker stepped in its activation, since it had not arrived at its goal before."""
    del cell, model
    return isinstance(walker, WalkerView) and not walker.arrived


def _get_footprint(walker: AgentViewProtocol, cell: object, model: ModelViewProtocol) -> float:
    """Return the footprint `intensity (1 - G / G_max)` a walker leaves on the ground of its cell."""
    if not isinstance(walker, WalkerView):
        return 0.0
    return walker.intensity * (1 - model.field('target.G', cell) / walker.G_max)


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
    walkers = GroupElement(
        'walkers',
        element=WalkerElement,
        size=n_walkers,
        values={
            'goal_x': np.array([float(goal[0]) for goal in goals]),
            'goal_y': np.array([float(goal[1]) for goal in goals]),
            'attraction': attraction,
            'width': float(size),
            'height': float(size),
        },
        location=[corners[start] for start in starts],
    )
    ground = FieldElement(
        'ground',
        element=GroundElement,
        space=LatticeSpace((size, size), boundary='reflecting'),
        values={'T': T},
    )
    return System(
        systems=[
            System('people', elements=[walkers], space=LatticeSpace((size, size), boundary='fixed')),
            System(
                'land',
                elements=[ground],
                interactions=[
                    RelaxationInteraction('fading', target='ground', state='G', time=_get_durability),
                    DiffusionInteraction('spreading', target='ground', state='V', coefficient=_get_diffusion),
                    RelaxationInteraction(
                        'smoothing',
                        target='ground',
                        state='V',
                        goal=_get_structure,
                        time=_get_relaxation,
                    ),
                ],
            ),
        ],
        interactions=[
            MovementInteraction('walk', target='people/walkers', step=_get_step, condition=_walks, schedule='member'),
            TransferInteraction(
                'footprints',
                source='people/walkers',
                target='land/ground',
                state=Source(),
                into='G',
                amount=_get_footprint,
                pairing='site',
                guard=_steps,
                form='agent',
                schedule='member',
            ),
            LifeCycleInteraction(
                'renewal',
                target='people/walkers',
                death=_waits,
                renew=True,
                values=_get_goal,
                schedule='member',
            ),
            MapInteraction(
                'arrival',
                target='people/walkers',
                state='arrived',
                function=_get_arrived,
                form='agent',
                schedule='member',
            ),
        ],
    )
