"""Define the compiled agents spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec
from ...specs._visual_spec import VisualSpec
from ..elements import Element, GroupElement
from ..laws import BaseLaw


@dataclass(frozen=True)
class CompiledAgentsSpec(BaseSpec):
    """Agent element or agent group of an agent-based problem.

    Its states live in the Mesa system.

    Args:
        path:
            The slash-separated path of the agents in the system tree.

        element:
            The agent element or the agent group.

        size:
            The initial number of agents.

        states:
            The names of the states every agent holds.

        values:
            A mapping from state name to its initial per-agent array or shared value.

        parameters:
            A mapping from parameter name to dotted name in the parameter vector.

        rules:
            The rules every agent runs each step, in declaration order.

        visual:
            The visual that draws the agents.

        objects:
            The names of the states that hold hashable objects.

        roles:
            A mapping from the name of a state with a role in a snapshot to its role.
    """

    path: str
    element: Element | GroupElement
    size: int
    states: tuple[str, ...]
    values: Mapping[str, object]
    parameters: Mapping[str, str]
    rules: tuple[BaseLaw, ...]
    visual: VisualSpec
    objects: tuple[str, ...] = ()
    roles: Mapping[str, str] = field(default_factory=dict)
