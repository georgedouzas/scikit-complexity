"""Define the compiled element spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec
from ...specs._variable_spec import VariableSpec
from ...specs._visual_spec import VisualSpec
from ..elements import Element, GroupElement


@dataclass(frozen=True)
class CompiledElementSpec(BaseSpec):
    """Element or group of a compiled problem with the dotted names of its variables.

    Args:
        path:
            The slash-separated path of the element or group in the system tree.

        element:
            The element or group.

        states:
            A mapping from state name to dotted name.

        parameters:
            A mapping from parameter name to dotted name.

        visual:
            The visual that draws the element or group.

        members:
            The size of a group, or `None` for a single element.

        adaptive:
            The declarations of the adaptive parameters, states that a schedule may set, by state name.

        lattice_shape:
            The number of sites along every dimension of the lattice the element or group lives on, or `None` off a
            lattice.

        roles:
            The roles of the states that declare one, by state name.
    """

    path: str
    element: Element | GroupElement
    states: Mapping[str, str]
    parameters: Mapping[str, str]
    visual: VisualSpec
    members: int | None = None
    adaptive: Mapping[str, VariableSpec] = field(default_factory=dict)
    lattice_shape: tuple[int, ...] | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
