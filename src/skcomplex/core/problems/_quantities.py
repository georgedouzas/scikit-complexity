"""Define the compiled flows, recorded values, and feedback terms of a system with conserved quantities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..base import Expression


@dataclass(frozen=True)
class _CompiledFlow:
    """Transfer of a compiled system between two holders."""

    path: str
    source: str | None
    sink: str | None
    source_label: str
    sink_label: str
    quantity: str | None


@dataclass(frozen=True)
class _Recorded:
    """Expression a trajectory records at every frame, with the references of its symbols."""

    path: str
    expression: Expression
    references: Mapping[Expression, Any]


@dataclass(frozen=True)
class _CompiledDelay:
    """Delay or smoothing of a compiled system, its call in the laws and its output read from hidden stocks."""

    call: Expression
    output: Expression
    references: Mapping[Expression, Any]


@dataclass(frozen=True)
class _CompiledObjective:
    """Objective of a compiled system, with its running and terminal values and the references of their symbols."""

    path: str
    expression: Expression
    terminal: Expression
    sense: str
    references: Mapping[Expression, Any]


@dataclass(frozen=True)
class LoopTerm:
    """Node of the feedback graph of a system with the expression its value or its rate is computed from.

    Args:
        kind:
            `'flow'`, or `'rate'` for a rate law outside flows.

        path:
            The node of a flow, or the state a rate law targets.

        expression:
            The expression.

        references:
            The references of the expression's symbols.

        source:
            The state a flow drains, or `None`.

        sink:
            The state a flow fills, or `None`.
    """

    kind: str
    path: str
    expression: Expression
    references: Mapping[Expression, Any]
    source: str | None = None
    sink: str | None = None


@dataclass(frozen=True)
class StockFlow:
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

        objectives:
            The objectives that optimal control solvers optimise.

        delays:
            The delays and smoothings of the laws, in the order the compiler replaced them by hidden stocks.
    """

    quantities: Mapping[str, tuple[str, ...]]
    flows: tuple[_CompiledFlow, ...] = ()
    recorded: tuple[_Recorded, ...] = ()
    boundary: Mapping[str, str] = field(default_factory=dict)
    terms: tuple[LoopTerm, ...] = ()
    discrete: bool = False
    objectives: tuple[_CompiledObjective, ...] = ()
    delays: tuple[_CompiledDelay, ...] = ()
