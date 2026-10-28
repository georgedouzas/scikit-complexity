"""Flatten a system tree into paths and validate its labels and values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.elements import Element, GroupElement
from ...core.interactions import BaseInteraction
from ...core.spaces import BaseSpace, LatticeSpace
from ...core.specs import WienerProcessSpec
from ...core.system import System
from ..problems import default_of

_LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
_MAX_SHOWN_COMPONENTS = 10


@dataclass(frozen=True)
class Node:
    """Element or interaction with its place in the system tree.

    Args:
        path:
            The slash-separated path of the component.

        scope:
            The path prefix of the system that holds the component.

        component:
            The element, group, or interaction, or the Wiener processes that the noise role of an interaction binds.

        space:
            The nearest enclosing space.

        agent:
            Whether the component runs on the agent stepper, as a pass of a formalism decides.

        active:
            The initial number of members of a variable group on the sites or edges of a network, whose capacity is
            its size, or `None` for every other component.
    """

    path: str
    scope: str
    component: Element | GroupElement | BaseInteraction | WienerProcessSpec
    space: BaseSpace | None
    agent: bool = False
    active: int | None = None

    @property
    def members(self) -> int | None:
        """Size of a group, or `None` for a single element or an interaction."""
        return self.component.size_ if isinstance(self.component, GroupElement) else None

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group, which runs on the agent stepper."""
        return self.agent

    @property
    def component_type(self) -> type[Element] | type[BaseInteraction]:
        """Element type of a group, or the type of an element or interaction."""
        return self.component.element if isinstance(self.component, GroupElement) else type(self.component)

    def value_of(self, variable: str) -> Any:  # noqa: ANN401
        """Return the value of a variable, or the element type's default for a group that does not set it.

        Args:
            variable:
                The name of the variable.

        Returns:
            value:
                The value given by the user or the default.
        """
        if not isinstance(self.component, GroupElement):
            return (
                getattr(self.component, variable)
                if hasattr(self.component, variable)
                else default_of(self.component, variable)
            )
        values = self.component.values_
        default = default_of(self.component.element, variable)
        if variable not in values:
            return default
        value = values[variable]
        space = getattr(self.component, 'space', None)
        if isinstance(space, LatticeSpace) and np.shape(value) == space.shape_:
            return np.ravel(value)
        if self.active is not None and np.ndim(value) and len(value) == self.active:
            rows = (self.members or 0) - self.active
            padding = np.broadcast_to(np.asarray(default, dtype=float), (rows, *np.shape(value)[1:]))
            return np.concatenate([np.asarray(value, dtype=float), padding])
        return value

    def _is_interaction(self) -> bool:
        """Check whether a node is an interaction."""
        return not isinstance(self.component, Element | GroupElement)


def _declared_times(system: System, scope: str = '', inherited: type | None = None) -> dict[str, type | None]:
    """Return the class of the declared time of every system of a tree by path, a nested system inheriting its parent's.

    Args:
        system:
            The root of the tree.

        scope:
            The path of the system.

        inherited:
            The time of the parent system.

    Returns:
        times:
            The time class of every system by path, such as `ContinuousTime`, or `None` for a system without one.
    """
    time = type(system.time) if system.time is not None else inherited
    times = {scope: time}
    for child in system.systems:
        times.update(_declared_times(child, f'{scope}/{child.label}' if scope else child.label, time))
    return times


