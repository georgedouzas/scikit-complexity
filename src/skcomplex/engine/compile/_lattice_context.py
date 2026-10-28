"""What lattice laws are compiled against: the lattice of every state and the symbols a component reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ...core.errors import ValidationError
from ..problems._arguments import Arguments
from ..problems._lattice_state import LatticeState


@dataclass
class LatticeContext:
    """What lattice laws are compiled against: the lattice of every state and the symbols a component reads.

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
    states: Mapping[Expression, LatticeState]
    arguments: Mapping[Expression, tuple[str, bool, int, int]]
    fused: bool = False
    others: Mapping[Expression, str] = field(default_factory=dict)

    def state(self, item: Expression, law: str) -> LatticeState:
        """Return the lattice state of a target symbol, or raise an error naming the law."""
        if item not in self.states:
            error_msg = f'{law} of {self.owner} targets {item}, which is not a state of a lattice group.'
            raise ValidationError(error_msg)
        return self.states[item]

    def keywords(self, function: Callable[..., Any]) -> Arguments:
        """Return the parameters a function accepts by keyword."""
        by_name = {
            name: (is_state, start, size) for name, is_state, start, size in self.arguments.values() if not is_state
        }
        names = self._accepted(function, list(by_name))
        return Arguments(
            names,
            tuple(by_name[name][0] for name in names),
            tuple(by_name[name][1] for name in names),
            tuple(by_name[name][2] for name in names),
        )

    @staticmethod
    def _accepted(function: Callable[..., Any], names: Sequence[str]) -> tuple[str, ...]:
        """Return the keyword names a function accepts among some names."""
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return tuple(names)
        parameters = signature.parameters.values()
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters):
            return tuple(names)
        keywords = {item.name for item in parameters if item.kind in {item.KEYWORD_ONLY, item.POSITIONAL_OR_KEYWORD}}
        return tuple(name for name in names if name in keywords)
