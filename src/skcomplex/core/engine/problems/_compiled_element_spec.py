"""Define the spec of an element of a compiled problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...base import BaseSpec
from ...components.elements import Element, VisualSpec
from ._group_element import _GroupElement
from ._variable_spec import _VariableSpec


@dataclass(frozen=True)
class CompiledElementSpec(BaseSpec):
    """A spec of an element or group of a compiled problem, with its variable names.

    Args:
        path:
            The slash-separated path of the element or group in the system tree.

        element:
            The element or group.

        states:
            A mapping from state name to dotted name.

        parameters:
            A mapping from parameter name to dotted name.

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
    element: Element | _GroupElement
    states: Mapping[str, str]
    parameters: Mapping[str, str]
    visual: VisualSpec
    members: int | None = None
    adaptive: Mapping[str, _VariableSpec] = field(default_factory=dict)
    lattice_shape: tuple[int, ...] | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
