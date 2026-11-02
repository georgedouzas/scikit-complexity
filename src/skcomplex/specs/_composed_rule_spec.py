"""Define the composed rule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import IntArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _ComposedRuleSpec(BaseSpec):
    """Rule of the composition of two automata, on the composite neighbourhood.

    Args:
        first:
            The local map of the automaton applied first.

        second:
            The local map of the automaton applied second.

        columns:
            For every cell the second rule reads, the cells of the composite neighbourhood the first rule reads there.

        states:
            The number of states of both automata.
    """

    first: IntArray
    second: IntArray
    columns: tuple[tuple[int, ...], ...]
    states: int
