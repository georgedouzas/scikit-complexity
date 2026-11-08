"""Define the loop spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _LoopSpec(BaseSpec):
    """Feedback loop.

    It holds its nodes from its first state, its polarity, and its gain at the point of evaluation.
    """

    nodes: tuple[str, ...]
    polarity: str
    gain: float
