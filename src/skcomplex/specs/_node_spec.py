"""Define the node spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._base_spec import BaseSpec
from ._wiener_process_spec import WienerProcessSpec

if TYPE_CHECKING:
    from ..core.elements import Element, GroupElement
    from ..core.interactions import BaseInteraction
    from ..core.spaces import BaseSpace


@dataclass(frozen=True)
class NodeSpec(BaseSpec):
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

        component_type:
            The element type of a group, or the type of an element, an interaction, or a Wiener process.

        members:
            The size of a group, or `None` for a single element, an interaction, or a Wiener process.

        interaction:
            Whether the component is not an element or a group: an interaction or a Wiener process.

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
    component_type: type
    members: int | None = None
    interaction: bool = False
    agent: bool = False
    active: int | None = None

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group, which runs on the agent stepper."""
        return self.agent
