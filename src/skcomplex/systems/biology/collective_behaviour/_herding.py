"""A shepherd that herds a flock of sheep to a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class HerdedSheepView(AgentViewProtocol, Protocol):
    """View of a herded sheep, with its states as attributes.

    The rules of `HerdedSheepElement` read them with checked types.
    """

    hx: float
    hy: float


class HerdedSheepElement(Element):
    """Sheep of a herded flock, an agent with a heading.

    A shepherding interaction moves it.

    Args:
        label:
            The label of the sheep or of the flock.

        hx:
            The first component of the heading of the last move.

        hy:
            The second component of the heading of the last move.

        values:
            Further values by name, such as the coordinates `x` and `y` of the sheep, or `None`.

        size:
            The number of sheep, or `None` for one.

        space:
            The pasture the sheep move on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import HerdedSheepElement
        >>> HerdedSheepElement('sheep', hx=1.0).hx
        1.0
    """

    visual = VisualSpec('agents', {})

    def __init__(
        self,
        label: str,
        *,
        hx: float | Array = 0.0,
        hy: float | Array = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.hx = hx
        self.hy = hy


class ShepherdView(AgentViewProtocol, Protocol):
    """View of a shepherd, with its states as attributes.

    The rules of `ShepherdElement` read them with checked types.
    """

    driving: float


class ShepherdElement(Element):
    """Shepherd of a flock, an agent with a mode.

    The mode records whether it collects stray sheep or drives the flock.

    Args:
        label:
            The label of the shepherd or of the group of shepherds.

        driving:
            1 while the shepherd drives the flock and 0 while it collects a stray sheep.

        values:
            Further values by name, such as the coordinates `x` and `y` of the shepherd, or `None`.

        size:
            The number of shepherds, or `None` for one.

        space:
            The pasture the shepherds move on, whose axes `x` and `y` place them, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShepherdElement
        >>> ShepherdElement('dog').driving
        0.0
    """

    visual = VisualSpec('agents', {})

    def __init__(
        self,
        label: str,
        *,
        driving: Annotated[float | Array, FiniteEnumeratedSet([0, 1])] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.driving = driving


class ShepherdingInteraction(Interaction):
    """Herding of a flock towards a goal by a shepherd.

    The shepherd collects stray sheep and drives the flock from behind.

    Following Strömbom et al. (2014), a sheep within `rs` of the shepherd moves the distance `delta` along `h H + c C +
    rho_a R_a + rho_s R_s + e E`, of unit vectors: its last heading `H`, the attraction `C` to the centre of its `n`
    nearest neighbours, the repulsion `R_a` from the sheep closer than `ra`, the repulsion `R_s` from the shepherd, and
    noise `E`. A sheep far from the shepherd grazes, a random step with the probability `graze`, and is repelled by
    close sheep. The shepherd collects the sheep furthest from the centre of the flock once it is further than `ra
    N^(2/3)`, heading to the point `ra` behind it, and otherwise drives the flock from the point `ra sqrt(N)` behind
    its centre, opposite the goal. It moves at the speed `delta_s` with noise, and stands still within `3 ra` of a
    sheep.

    Args:
        label:
            The label of the interaction.

        sheep:
            The path of the flock.

        shepherds:
            The path of the group of the shepherd.

        goal:
            The position of the goal.

        ra:
            The distance of repulsion between sheep.

        rs:
            The distance at which sheep detect the shepherd.

        n:
            The number of nearest neighbours whose centre attracts a sheep.

        weights:
            The weights `(h, c, rho_a, rho_s, e)` of the heading, the attraction, the two repulsions, and the noise.

        delta:
            The step of a sheep.

        delta_s:
            The step of the shepherd.

        graze:
            The probability of a grazing step.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShepherdingInteraction
        >>> ShepherdingInteraction('herding', sheep='sheep', shepherds='dogs').rs
        65.0
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        sheep: Role,
        shepherds: Role,
        goal: Sequence[float] = (0.0, 0.0),
        ra: float = 2.0,
        rs: float = 65.0,
        n: int = 20,
        weights: Sequence[float] = (0.5, 1.05, 2.0, 1.0, 0.3),
        delta: float = 1.0,
        delta_s: float = 1.5,
        graze: float = 0.05,
    ) -> None:
        super().__init__(label)
        self.sheep = sheep
        self.shepherds = shepherds
        self.goal = goal
        self.ra = ra
        self.rs = rs
        self.n = n
        self.weights = weights
        self.delta = delta
        self.delta_s = delta_s
        self.graze = graze

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the herding rule.

        Args:
            context:
                The namespace of the symbols the laws can use.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self.herd)]

    def herd(self, system: SystemView) -> None:
        """Move every sheep and every shepherd from the positions at the start of the step.

        Args:
            system:
                The system.
        """
        flock = cast(list[HerdedSheepView], system.get_agents(self.sheep))
        dogs = cast(list[ShepherdView], system.get_agents(self.shepherds))
        sheep = np.array([np.asarray(item.location, dtype=float) for item in flock])
        shepherds = np.array([np.asarray(item.location, dtype=float) for item in dogs])
        headings = np.array([[item.hx, item.hy] for item in flock])
        moved, headings = self._move_sheep(sheep, shepherds, headings, system.rng)
        targets, driving = self._get_targets(sheep, shepherds)
        steps = self._get_unit(targets - shepherds) + 0.3 * self._get_unit(system.rng.normal(size=shepherds.shape))
        distances = np.linalg.norm(sheep[None, :, :] - shepherds[:, None, :], axis=2)
        blocked = distances.min(axis=1) < 3 * self.ra
        shepherds = shepherds + np.where(blocked[:, None], 0.0, self.delta_s * self._get_unit(steps))
        width, height = system.dimensions or (np.inf, np.inf)
        for sheep_view, position, heading in zip(flock, moved, headings, strict=True):
            sheep_view.move_to(self._get_inside(position, width, height))
            sheep_view.hx, sheep_view.hy = float(heading[0]), float(heading[1])
        for dog, position, mode in zip(dogs, shepherds, driving, strict=True):
            dog.move_to(self._get_inside(position, width, height))
            dog.driving = float(mode)

    def _move_sheep(
        self,
        sheep: Array,
        shepherds: Array,
        headings: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, Array]:
        """Return the new positions and headings of the sheep."""
        h, c, rho_a, rho_s, e = self.weights
        offsets = sheep[:, None, :] - sheep[None, :, :]
        distances = np.linalg.norm(offsets, axis=2)
        np.fill_diagonal(distances, np.inf)
        close = distances < self.ra
        repulsion = self._get_unit((offsets * close[:, :, None] / np.maximum(distances, 1e-9)[:, :, None]).sum(axis=1))
        nearest = np.argsort(distances, axis=1)[:, : self.n]
        attraction = self._get_unit(sheep[nearest].mean(axis=1) - sheep)
        dog_offsets = sheep[:, None, :] - shepherds[None, :, :]
        dog_distances = np.linalg.norm(dog_offsets, axis=2)
        alert = dog_distances.min(axis=1) < self.rs
        fear = self._get_unit((dog_offsets / np.maximum(dog_distances, 1e-9)[:, :, None]).sum(axis=1))
        noise = self._get_unit(rng.normal(size=sheep.shape))
        herded = h * self._get_unit(headings) + c * attraction + rho_a * repulsion + rho_s * fear + e * noise
        grazing = rho_a * repulsion + (rng.random(len(sheep)) < self.graze)[:, None] * noise
        heading = np.where(alert[:, None], self._get_unit(herded), grazing)
        return sheep + self.delta * heading, heading

    def _get_targets(self, sheep: Array, shepherds: Array) -> tuple[Array, Array]:
        """Return the points the shepherds head to and whether each drives the flock."""
        centre, goal = sheep.mean(axis=0), np.asarray(self.goal, dtype=float)
        spread = np.linalg.norm(sheep - centre, axis=1)
        stray = int(np.argmax(spread))
        if spread[stray] > self.ra * len(sheep) ** (2 / 3):
            target = sheep[stray] + self.ra * self._get_unit((sheep[stray] - centre)[None, :])[0]
            return np.broadcast_to(target, shepherds.shape), np.zeros(len(shepherds), dtype=bool)
        target = centre + self.ra * np.sqrt(len(sheep)) * self._get_unit((centre - goal)[None, :])[0]
        return np.broadcast_to(target, shepherds.shape), np.ones(len(shepherds), dtype=bool)

    def _get_unit(self, vectors: Array) -> Array:
        """Return the vectors scaled to unit length.

        Zero vectors stay unchanged.
        """
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.where(lengths > 0, lengths, 1.0)

    def _get_inside(self, position: Array, width: float, height: float) -> tuple[float, float]:
        """Return a position clipped to the field."""
        margin = 1e-6
        return float(np.clip(position[0], margin, width - margin)), float(np.clip(position[1], margin, height - margin))


class ShepherdingSystem(BaseSystem):
    """A shepherd that collects a scattered flock and drives it to a goal.

    It is the heuristic of Strömbom et al.

    The sheep start scattered in the middle of a square field and the shepherd behind them, opposite the goal. Sheep
    flee a near shepherd while they keep together and apart, and graze otherwise. The shepherd alternates between
    collecting the sheep furthest from the flock and driving the flock from behind towards the goal, and the zig-zag of
    this alternation brings the flock to the goal in a few hundred steps. The system has the agent groups `sheep` and
    `shepherds`, the latter of one shepherd.

    Args:
        n_sheep:
            The number of sheep.

        size:
            The side of the square field.

        goal:
            The position of the goal.

        seed:
            The seed of the initial positions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShepherdingSystem
        >>> [group.label for group in ShepherdingSystem(10).prepare().system_.elements]
        ['sheep', 'shepherds']
    """

    field = 'biology/collective_behaviour'
    topics = ('shepherding', 'herding', 'agents', 'swarm control')
    references = ('Strömbom et al. 2014', 'Long et al. 2020')
    until = 300

    def __init__(
        self,
        n_sheep: int = 30,
        *,
        size: float = 100.0,
        goal: tuple[float, float] = (5.0, 5.0),
        seed: int | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sheep = n_sheep
        self.size = size
        self.goal = goal
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a shepherd that drives a scattered flock to a goal.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        positions = rng.uniform(0.4 * self.size, 0.7 * self.size, (self.n_sheep, 2))
        space = ContinuousSpace(extent=(self.size, self.size))
        sheep = HerdedSheepElement(
            'sheep',
            values={'x': positions[:, 0], 'y': positions[:, 1]},
            size=self.n_sheep,
            space=space,
        )
        dogs = ShepherdElement(
            'shepherds',
            values={'x': [0.9 * self.size], 'y': [0.9 * self.size]},
            size=1,
            space=space,
        )
        herding = ShepherdingInteraction(
            'herding',
            sheep='sheep',
            shepherds='shepherds',
            goal=self.goal,
            n=max(1, round(0.9 * self.n_sheep)),
        )
        self.system_ = System(
            self.label,
            elements=[sheep, dogs],
            interactions=[herding],
            space=space,
        )
