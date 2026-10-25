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
            `'auxiliary'`, `'flow'`, or `'rate'` for a rate law outside flows.

        path:
            The node of an auxiliary or a flow, or the state a rate law targets.

        expression:
            The expression, with the symbols of auxiliaries kept.

        auxiliaries:
            The paths of the auxiliaries by their symbols.

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
    auxiliaries: Mapping[Expression, str]
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
            The flows and auxiliaries that trajectories record.

        boundary:
            The hidden state that accumulates the net inflow from the environment of each quantity with boundary
            flows.

        terms:
            The terms of the feedback graph.

        discrete:
            Whether flow amounts are per step rather than rates.

        invariants:
            The tolerance of every invariant by the path that trajectories record it under.

        objectives:
            The objectives that optimal control solvers optimise.
    """

    quantities: Mapping[str, tuple[str, ...]]
    flows: tuple[_CompiledFlow, ...] = ()
    recorded: tuple[_Recorded, ...] = ()
    boundary: Mapping[str, str] = field(default_factory=dict)
    terms: tuple[LoopTerm, ...] = ()
    discrete: bool = False
    invariants: Mapping[str, float] = field(default_factory=dict)
    objectives: tuple[_CompiledObjective, ...] = ()
