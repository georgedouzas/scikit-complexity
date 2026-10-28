"""Define the compiled element spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._base_spec import BaseSpec
from ._visual_spec import VisualSpec

if TYPE_CHECKING:
    from ..core.elements import Element, GroupElement


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
    """

    path: str
    element: Element | GroupElement
    states: Mapping[str, str]
    parameters: Mapping[str, str]
    visual: VisualSpec
    members: int | None = None
