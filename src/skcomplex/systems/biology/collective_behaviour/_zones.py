"""Define zone coupling.

Agents couple to the other agents within priority zones around them, as fish and birds in a school or a flock.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
import numpy.typing as npt

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, RoleSpec, Tags
from ....core.types import Array, ElementKind

REDUCTIONS = ('sum', 'mean')

type AgentCombination = Callable[[AgentViewProtocol, SystemViewProtocol, Array | None, int | None], npt.ArrayLike]


@dataclass(frozen=True, eq=False)
class RadiusSpec(BaseSpec):
    """Term of the other agents within a radius of an agent.

    It is reduced by a sum or a mean, with a priority. Every other agent within `radius` gives the vector
    `function(agent, other, system)`, and the terms reduce by `reduce`. Among several radii, only the radii of the
    highest priority that hold another agent count.

    Args:
        radius:
            The radius, a number or the name of a parameter of the agent.

        function:
            The term of one other agent, a function of the agent, the other agent, and the system view that returns a
            vector.

        reduce:
            `'sum'` or `'mean'`.

        priority:
            The priority of the radius, higher first.

    Raises:
        ValidationError:
            The reduction is unknown, the function is not callable, or the radius is negative, raised when its owner
            checks it.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import RadiusSpec
        >>> RadiusSpec(2.0, lambda agent, other, system: (1.0, 0.0), reduce='mean').reduce
        'mean'
    """

    radius: float | str
    function: Callable[..., object]
    reduce: str = 'sum'
    priority: int = 0

    def _check(self) -> None:
        """Raise an error for an unknown reduction, a bad function, or a negative radius.

        The function must be callable.

        Raises:
            ValidationError:
                The reduction is unknown, the function is not callable, or the radius is negative.
        """
        if self.reduce not in REDUCTIONS:
            error_msg = f'The reduction of a radius is {self.reduce!r}. Use one of {", ".join(REDUCTIONS)}.'
            raise ValidationError(error_msg)
        if not callable(self.function):
            error_msg = f'The function of a radius must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.radius, str) and not self.radius >= 0:
            error_msg = f'The radius is {self.radius!r}. Give a nonnegative number or the name of a parameter.'
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class ClampSpec(BaseSpec):
    """Clamp of the change of a vector.

    It bounds the largest angle the vector turns and its norm. Every bound is a number, the name of a parameter of the
    agent, or `None` for no bound. The vector turns from its current direction toward its new one by at most
    `max_angle`, keeping the new norm, and its norm is then clipped between `min_norm` and `max_norm`. A zero vector
    stays zero.

    Args:
        max_angle:
            The largest angle in radians the vector turns, or `None`.

        min_norm:
            The smallest norm, or `None`.

        max_norm:
            The largest norm, or `None`.

    Raises:
        ValidationError:
            A bound is negative, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ClampSpec
        >>> ClampSpec(min_norm=0.5, max_norm='speed').max_norm
        'speed'
    """

    max_angle: float | str | None = None
    min_norm: float | str | None = None
    max_norm: float | str | None = None

    def _check(self) -> None:
        """Raise an error for a negative bound.

        Raises:
            ValidationError:
                A bound is negative.
        """
        for name, value in (('max_angle', self.max_angle), ('min_norm', self.min_norm), ('max_norm', self.max_norm)):
            if value is not None and not isinstance(value, str) and not value >= 0:
                error_msg = f'The bound {name} of a clamp is {value!r}. Give a nonnegative number, a name, or None.'
                raise ValidationError(error_msg)


class ZoneCouplingInteraction(CouplingInteraction):
    """Coupling of every agent of a group to the other agents within priority zones.

    Every agent runs the coupling in its own activation in the phase `phase`. Every `RadiusSpec` of `radii` gives the
    vector `function(agent, other, system)` of every other agent of the source group, or of the target group, within
    its radius, reduced by its sum or mean. Only the radii of the highest priority that hold another agent count, and
    their reductions add up. The states `write` of the agent form a vector `v`, which becomes
    `combine(agent, system, reduction, priority)`, or `v` plus the reduction by default, where `reduction` and
    `priority` are `None` when no radius holds another agent. `clamp` then bounds the angle the vector turns and its
    norm.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of the agents.

        source:
            The path of the group of the other agents, or `None` for the agents of the target.

        write:
            The names of the components of the vector of the agent.

        combine:
            The next vector, a callable of the agent, the system view, the reduction, and its priority, or `None` for
            the vector plus the reduction.

        radii:
            The terms over the radii around an agent.

        clamp:
            The clamp of the turn and the norm of the vector, or `None`.

        phase:
            The phase of the step the agents run in, or `None` for the default phase.

    Raises:
        ValidationError:
            The target is not one group, there is no radius, or a radius or the clamp is invalid, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import RadiusSpec, ZoneCouplingInteraction
        >>> term = RadiusSpec(1.0, lambda agent, other, system: (0, 0))
        >>> zones = ZoneCouplingInteraction('zones', target='fish', write=('vx', 'vy'), radii=(term,))
        >>> zones.prepare().radii[0].radius
        1.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('write',))],
        source: Annotated[str | None, RoleSpec(optional=True, elements=(ElementKind.ANY,))] = None,
        write: str | Sequence[str] = 'x',
        combine: AgentCombination | None = None,
        radii: Sequence[RadiusSpec] = (),
        clamp: ClampSpec | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, source=source, write=write)
        self.combine = combine
        self.radii = radii
        self.clamp = clamp
        self.phase = phase

    def _get_tags(self) -> Tags:
        """Return the tags of the agent rule."""
        return self._get_variant_tags((RuleLaw,))

    def _check_form(self) -> None:
        """Raise an error for a target that is not one group, no radii, or an invalid radius or clamp."""
        if not self.radii or not isinstance(self.target, str):
            error_msg = f'The zone coupling {self.label!r} needs one group of agents and at least one radius.'
            raise ValidationError(error_msg)
        for spec in self.radii:
            spec._check()
        if self.clamp is not None:
            self.clamp._check()

    @staticmethod
    def _get_bound(agent: AgentViewProtocol, bound: float | str) -> float:
        """Return a bound given as a number or as the name of a parameter of the agent."""
        return float(cast(float, getattr(agent, bound))) if isinstance(bound, str) else float(bound)

    def _get_agent_reduction(
        self,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
    ) -> tuple[Array | None, int | None]:
        """Return the sum of the reduced terms of the top-priority radii holding an agent.

        The radii are those of the highest priority that hold another agent.
        """
        groups = set(system.get_groups(self._get_role('source' if self.source is not None else 'target')))
        found: dict[float, list[AgentViewProtocol]] = {}
        levels: dict[int, list[Array]] = {}
        for spec in self.radii:
            radius = self._get_bound(agent, spec.radius)
            if radius not in found:
                found[radius] = [other for other in agent.get_neighbors(radius=radius) if other.group in groups]
            others = found[radius]
            if not others:
                continue
            terms = np.array([np.asarray(spec.function(agent, other, system), dtype=float) for other in others])
            levels.setdefault(spec.priority, []).append(
                terms.mean(axis=0) if spec.reduce == 'mean' else terms.sum(axis=0),
            )
        if not levels:
            return None, None
        priority = max(levels)
        return np.sum(levels[priority], axis=0), priority

    @staticmethod
    def _turn(current: Array, value: Array, max_angle: float) -> Array:
        """Return a vector turned from the current direction toward its own by an angle.

        The turn is at most the angle and keeps the norm.
        """
        norm, size = float(np.linalg.norm(value)), float(np.linalg.norm(current))
        if norm == 0 or size == 0:
            return value
        direction, facing = value / norm, current / size
        angle = float(np.arccos(np.clip(facing @ direction, -1.0, 1.0)))
        if angle <= max_angle:
            return value
        if angle == np.pi:
            direction = np.eye(len(facing))[int(np.argmin(np.abs(facing)))]
        normal = direction - (facing @ direction) * facing
        normal = normal / np.linalg.norm(normal)
        return norm * (np.cos(max_angle) * facing + np.sin(max_angle) * normal)

    def _get_clamped(self, agent: AgentViewProtocol, current: Array, value: Array) -> Array:
        """Return a new vector with its turn and its norm clamped."""
        clamp = self.clamp
        if clamp is None:
            return value
        if clamp.max_angle is not None:
            value = self._turn(current, value, self._get_bound(agent, clamp.max_angle))
        norm = float(np.linalg.norm(value))
        if norm == 0:
            return value
        low = 0.0 if clamp.min_norm is None else self._get_bound(agent, clamp.min_norm)
        high = np.inf if clamp.max_norm is None else self._get_bound(agent, clamp.max_norm)
        return value * min(max(norm, low), high) / norm

    def _update_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Set the written states of one agent to the clamped combination of the reduction.

        The reduction is over its radii.
        """
        if not agent.alive:
            return
        names = [self.write] if isinstance(self.write, str) else list(self.write)
        current = np.array([float(cast(float, getattr(agent, name))) for name in names])
        reduction, priority = self._get_agent_reduction(agent, system)
        if self.combine is not None:
            combine = cast(AgentCombination, self.combine)
            value = np.asarray(combine(agent, system, reduction, priority), dtype=float)
        else:
            value = current if reduction is None else current + reduction
        for name, item in zip(names, self._get_clamped(agent, current, value), strict=True):
            setattr(agent, name, float(item))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that updates every agent.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule of the agents.
        """
        return [RuleLaw(self._update_member, phase=self.phase, members=self._get_role('target'))]
