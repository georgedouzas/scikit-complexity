"""Define the spec of a part with its place in the tree of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ...specs._base_spec import BaseSpec
from ...specs._wiener_process_spec import WienerProcessSpec
from ..elements import BaseElement, Element, GroupElement
from ..interactions import BaseInteraction
from ..spaces import BaseSpace


@dataclass(frozen=True)
class NodeSpec(BaseSpec):
    """A spec of an element or interaction with its place in the tree of a system.

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
            What declares the variables of the component, which `declarations_of` reads: the element type of a group,
            or the type of an element, an interaction, or a Wiener process, or the element or a member of the group for
            elements that declare their variables from data.

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
    component_type: type | BaseElement
    members: int | None = None
    interaction: bool = False
    agent: bool = False
    active: int | None = None

    @property
    def as_element(self) -> Element | GroupElement:
        """The component of a node of an element or a group.

        Raises:
            TypeError:
                The component is an interaction or a Wiener process.
        """
        if not isinstance(self.component, Element | GroupElement):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not an element or a group.'
            raise TypeError(error_msg)
        return self.component

    @property
    def as_group(self) -> GroupElement:
        """The component of a node of a group.

        Raises:
            TypeError:
                The component is not a group.
        """
        if not isinstance(self.component, GroupElement):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not a group.'
            raise TypeError(error_msg)
        return self.component

    @property
    def as_interaction(self) -> BaseInteraction:
        """The component of a node of an interaction.

        Raises:
            TypeError:
                The component is an element, a group, or a Wiener process.
        """
        if not isinstance(self.component, BaseInteraction):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not an interaction.'
            raise TypeError(error_msg)
        return self.component

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group.

        Such a node runs on the agent stepper.
        """
        return self.agent
