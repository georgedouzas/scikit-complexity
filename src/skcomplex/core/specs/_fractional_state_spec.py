"""Define the fractional state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class FractionalStateSpec(BaseSpec):
    """A spec of a state of fractional order with the path of its part.

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
