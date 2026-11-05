"""Define the group spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...specs import BaseSpec
from ..types import Arrays


@dataclass(frozen=True)
class _GroupSpec(BaseSpec):
    """Productions of one predecessor, context, and condition, one of which a matching module takes."""

    code: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    condition: Callable[..., Arrays] | None
    variables: int
    successors: tuple[int, ...] = ()
    probabilities: tuple[Callable[..., Arrays] | None, ...] = ()
    rank: tuple[int, int, int] = (0, 0, 0)
