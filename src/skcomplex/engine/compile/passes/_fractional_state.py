"""State of a fractional order with the path of its component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FractionalState:
    """State of a fractional order with the path of its component.

    Args:
        path:
            The path of the element or group.

        name:
            The dotted name of the state.

        order:
            The order of the state.
    """

    path: str
    name: str
    order: float
