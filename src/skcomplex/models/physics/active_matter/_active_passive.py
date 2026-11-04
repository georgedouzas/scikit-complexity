"""Flocking of active agents among passive ones: self-propulsion, alignment, noise, and a soft repulsion between all."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, sqrt
from sage.modules.free_module_element import FreeModuleElement

from ....core.elements import GroupElement, PointElement
from ....core.interactions import PairCouplingInteraction, RateInteraction
from ....core.laws.utils import neighbors
from ....core.spaces import ContinuousSpace, RadiusNeighborhood
from ....core.system import System
from ....engine.backends import NumbaBackend
from ....engine.solvers import EulerMaruyamaSolver
from ....errors import ValidationError
from ....specs import ObservableSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

SOFTENING = 1e-12
HERTZIAN_POWER = 3.5
SMALLEST_GROUP = 2


def _grid(n_agents: int, size: float, rng: np.random.Generator) -> Array:
    """Return shuffled positions on a jittered square grid that fills a periodic box."""
    side = math.ceil(math.sqrt(n_agents))
    spacing = size / side
    cells = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)
    chosen = rng.permutation(len(cells))[:n_agents]
    return np.mod((cells[chosen] + 0.5) * spacing + rng.normal(0.0, 0.05 * spacing, (n_agents, 2)), size)


def _hertzian_energy(s: SimpleNamespace, r: Expression, *, epsilon: float, radius: float) -> Expression:
    """Return the energy whose force is the Hertzian repulsion `epsilon (1 - r / (2 r_0))^(5/2)`."""
    del s
    overlap = 1 - r / (2 * radius)
    return epsilon * 2 * radius * overlap**HERTZIAN_POWER / HERTZIAN_POWER


def _propulsion(s: SimpleNamespace, *, mu: float, v0: float) -> FreeModuleElement:
    """Return the force `mu (v0 - |v|) v / |v|` that brings a speed to `v0`."""
    velocity = s.target.x_d1
    speed = sqrt(velocity.dot_product(velocity) + SOFTENING)
    return velocity * (mu * (v0 - speed) / speed)


def _flocking(s: SimpleNamespace, *, alpha: float, reach: float) -> FreeModuleElement:
    """Return the force of size `alpha` along the mean velocity of the agents within a reach, the agent included."""
    mean = neighbors(s.target.other.x_d1, over=RadiusNeighborhood(reach, center=True), reduce='mean')
    return mean * (alpha / sqrt(mean.dot_product(mean) + SOFTENING))


def _velocity_x(member: SimpleNamespace) -> Expression:
    """Return the first component of the velocity of a member."""
    return member.x_d1[0]


def _velocity_y(member: SimpleNamespace) -> Expression:
    """Return the second component of the velocity of a member."""
    return member.x_d1[1]


def _speed(member: SimpleNamespace) -> Expression:
    """Return the speed of a member."""
    return sqrt(member.x_d1.dot_product(member.x_d1))


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean velocity of the active agents over their mean speed."""
    return sqrt(s.velocity_x**2 + s.velocity_y**2) / s.mean_speed


def _mean_distance(t: float, values: Mapping[str, object], *, group: str, size: float) -> float:
    """Return the mean distance between the members of a group in a periodic box, 0 for fewer than two members."""
    del t
    positions = np.asarray(values.get(f'{group}.x', np.zeros((0, 2))), dtype=float)
    count = len(positions)
    if count < SMALLEST_GROUP:
        return 0.0
    separations = positions[:, None, :] - positions[None, :, :]
    separations -= size * np.round(separations / size)
    distances: Array = np.sqrt((separations**2).sum(axis=-1))
    return float(distances.sum() / (count * (count - 1)))


def _observables(system: System) -> list[ObservableSpec]:
    """Return the velocity of the active agents, their polarization, and the mean distance in both groups."""
    if not isinstance(system.space, ContinuousSpace):
        error_msg = 'The observables of `active_passive_flocking` read the periodic box of its continuous space.'
        raise ValidationError(error_msg)
    size = float(system.space.extent[0])
    return [
        ObservableSpec('velocity_x', _velocity_x, group='active'),
        ObservableSpec('velocity_y', _velocity_y, group='active'),
        ObservableSpec('mean_speed', _speed, group='active'),
        ObservableSpec('polarization', _polarization),
        ObservableSpec('active_distance', function=functools.partial(_mean_distance, group='active', size=size)),
        ObservableSpec('passive_distance', function=functools.partial(_mean_distance, group='passive', size=size)),
    ]


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'flocking', 'phase separation', 'pair interaction'),
    references=('Silverberg et al. 2013', 'Charbonneau 2017'),
    defaults={'until': 100.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01), 'backend': NumbaBackend()},
    observables=_observables,
)
def active_passive_flocking(
    n_agents: int = 300,
    *,
    active_share: float = 0.3,
    packing_fraction: float = 0.94,
    alpha: float = 2.0,
    noise: float = 0.5,
    v0: float = 1.0,
    mu: float = 0.05,
    epsilon: float = 100.0,
    radius: float = 1.0,
    flocking_radius: float = 4.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the active and passive agents of Silverberg et al. (2013) in a periodic box.

    All agents repel each other when they overlap with the force `epsilon (1 - r / (2 r_0))^(5/2)`. Active agents also
    feel the propulsion `mu (v0 - |v|) v / |v|`, the alignment `alpha V / |V|` with the mean velocity `V` of the active
    agents within the flocking radius, and a Gaussian noise. Passive agents only slow down, `-mu v`. The velocities of
    the active agents order when alignment dominates the noise.

    Args:
        n_agents:
            The number of agents.

        active_share:
            The share of the agents that are active.

        packing_fraction:
            The share of the periodic box that the discs of radius `r_0` cover.

        alpha:
            The size of the alignment force, 0 for no flocking.

        noise:
            The amplitude of the noise of the active agents.

        v0:
            The preferred speed of the active agents.

        mu:
            The strength of the propulsion.

        epsilon:
            The strength of the repulsion.

        radius:
            The radius `r_0` of an agent, so agents overlap below the distance `2 r_0`.

        flocking_radius:
            The radius within which an active agent reads the velocities of the others.

        random_state:
            The seed or generator of the initial positions and velocities.

    Returns:
        system:
            The model, with the positions `active.x` and `passive.x`, the velocities `active.x_d1` and `passive.x_d1`,
            and the observables `polarization`, `active_distance`, and `passive_distance`.

    Raises:
        ValidationError:
            The shares leave no active agent.

    Examples:
        >>> from skcomplex.models.physics.active_matter import active_passive_flocking
        >>> [element.label for element in active_passive_flocking(20).elements]
        ['active', 'passive']
    """
    n_active = round(active_share * n_agents)
    if n_active < 1:
        error_msg = f'The active share {active_share} of {n_agents} agents leaves no active agent. Raise the share.'
        raise ValidationError(error_msg)
    size = math.sqrt(math.pi * radius**2 * n_agents / packing_fraction)
    rng = np.random.default_rng(random_state)
    grid = _grid(n_agents, size, rng)
    labels = {'active': grid[:n_active], 'passive': grid[n_active:]}
    velocities = {'active': v0, 'passive': 0.1 * v0}
    groups = [
        GroupElement(
            label,
            element=PointElement,
            size=len(positions),
            values={'x': positions, 'x_d1': rng.normal(0.0, velocities[label], (len(positions), 2))},
            visual=VisualSpec('points', {'position': 'x', 'radius': radius, 'states': ()}),
        )
        for label, positions in labels.items()
        if len(positions)
    ]
    names = [group.label for group in groups]
    energy = functools.partial(_hertzian_energy, epsilon=epsilon, radius=radius)
    repulsions = [
        PairCouplingInteraction(f'{first}_{second}', first, second, cutoff=2 * radius, potential=energy, period=size)
        for index, first in enumerate(names)
        for second in names[index:]
    ]
    rates = [
        RateInteraction(
            'active_propulsion',
            target='active',
            rate=functools.partial(_propulsion, mu=mu, v0=v0),
            noise=noise,
        ),
        RateInteraction(
            'flocking',
            target='active',
            rate=functools.partial(_flocking, alpha=alpha, reach=flocking_radius),
        ),
    ]
    if 'passive' in names:
        rates.append(
            RateInteraction('passive_damping', target='passive', rate=functools.partial(_propulsion, mu=mu, v0=0.0)),
        )
    return System(
        elements=groups,
        interactions=[*repulsions, *rates],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
