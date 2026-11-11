"""Define the spec of what the laws of a lattice are compiled against."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._lattice_state_spec import LatticeStateSpec


@dataclass(frozen=True)
class LatticeContextSpec(BaseSpec):
    """A spec of what the laws of a lattice are compiled against.

    It holds the lattice of every state and the symbols each law reads.

    Args:
        owner:
            The path of the component that declares the laws.

        states:
            The lattice state of every target symbol.

        arguments:
            The parameters and states the component reads, by symbol, with their keyword names.

        fused:
            Whether lookup rules run as compiled loops.

        others:
            The entry name of every symbol of a neighbour, the namespace `s.other` of the laws.
    """

    owner: str
    states: Mapping[Expression, LatticeStateSpec]
    arguments: Mapping[Expression, tuple[str, bool, int, int]]
    fused: bool = False
    others: Mapping[Expression, str] = field(default_factory=dict)
