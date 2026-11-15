"""Define the fractional state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class _FractionalStateSpec(BaseSpec):
    """A spec of a state of fractional order with the path of its component.

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
