"""Quantities, flows, and recorded values of a compiled system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ._base import BaseValue
from ._loop_term import LoopTerm


@dataclass(frozen=True)
class _CompiledFlow(BaseValue):
    """Transfer of a compiled system between two holders."""

    path: str
    source: str | None
    sink: str | None
    source_label: str
    sink_label: str
    quantity: str | None


@dataclass(frozen=True)
class _Recorded(BaseValue):
    """Expression a trajectory records at every frame, with the references of its symbols."""

    path: str
    expression: Expression
    references: Mapping[Expression, Any]


@dataclass(frozen=True)
class _CompiledDelay(BaseValue):
    """Delay or smoothing of a compiled system, its call in the laws and its output read from hidden states."""

    call: Expression
    output: Expression
    references: Mapping[Expression, Any]


@dataclass(frozen=True)
class TransferNetwork(BaseValue):
    """Quantities, flows, and recorded values of a compiled system.

    Args:
        quantities:
            The names of the states that hold each quantity.

        flows:
            The flows.

        recorded:
            The flows and the running values of objectives that trajectories record.

        boundary:
            The hidden state that accumulates the net inflow from the environment of each quantity with boundary
            flows.

        terms:
            The terms of the feedback graph.

        discrete:
            Whether flow amounts are per step rather than rates.

        delays:
            The delays and smoothings of the laws, in the order the compiler replaced them by hidden states.
    """

    quantities: Mapping[str, tuple[str, ...]]
    flows: tuple[_CompiledFlow, ...] = ()
    recorded: tuple[_Recorded, ...] = ()
    boundary: Mapping[str, str] = field(default_factory=dict)
    terms: tuple[LoopTerm, ...] = ()
    discrete: bool = False
    delays: tuple[_CompiledDelay, ...] = ()
