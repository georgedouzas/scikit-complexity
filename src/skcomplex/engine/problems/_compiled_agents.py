"""Agent element or agent group of an agent-based problem, whose states live in the Mesa model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ...core.elements import Element, GroupElement
from ...core.laws import BaseLaw
from ...specs import VisualSpec


@dataclass(frozen=True)
class CompiledAgents:
    """Agent element or agent group of an agent-based problem, whose states live in the Mesa model.

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
    """

    path: str
    element: Element | GroupElement
    size: int
    states: tuple[str, ...]
    values: Mapping[str, object]
    parameters: Mapping[str, str]
    rules: tuple[BaseLaw, ...]
    visual: VisualSpec
