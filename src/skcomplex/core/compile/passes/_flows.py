"""Compile the flows of stocks, auxiliaries, conserved quantities, and system-dynamics delays of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...laws import Auxiliary, Invariant, Law, Transfer
from .._census import Census
from .._compilation import BlockLaws, Compilation
from .._declarations import Declarations
from .._flows import StockFlowPass, quantities_of
from .._pipeline import Pass
from .._problem import Problem
from .._quantities import StockFlow
from .._specs import KernelSpec


def _flow_owner(spec: KernelSpec, law: Law, agents: set[str]) -> str | None:
    """Return the component that declares a flow, an auxiliary, or an invariant, or `None` for another law."""
    del agents
    return spec.owner if isinstance(law, Transfer | Auxiliary | Invariant) else None


class ExpandFlowsPass(Pass):
    """Pass that expands the flows, auxiliaries, invariants, and objectives of components into laws of other kinds."""

    reads = ('collected', 'layout')
    writes = ('collected', 'flows')

    def run(self, compilation: Compilation) -> None:
        """Expand the flows of the collected laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        flows = StockFlowPass(quantities_of(declarations.blocks))
        compilation.collected = [
            (spec, flows.expand(spec, list(laws)), block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]
        compilation.facts['flows'] = flows


class FlowsCensusPass(Pass):
    """Pass that leaves the auxiliaries and invariants, which change no state, out of the census."""

    reads = ('census',)
    writes = ('census',)

    def run(self, compilation: Compilation) -> None:
        """Leave the auxiliaries and invariants out of the census.

        Args:
            compilation:
                The compilation state.
        """
        census = cast('Census', compilation.census)
        records = [record for record in census.records if not isinstance(record.law, Auxiliary | Invariant)]
        compilation.census = replace(census, records=records)


class LowerFlowsPass(Pass):
    """Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

    reads = ('lowered', 'layout', 'flows')
    writes = ('lowered', 'stock_flow')

    def run(self, compilation: Compilation) -> None:
        """Lower the flows and their delays.

        Args:
            compilation:
                The compilation state.
        """
        flows = cast('StockFlowPass', compilation.facts['flows'])
        lowered = cast('BlockLaws', compilation.lowered)
        declarations = cast('Declarations', compilation.layout)
        compilation.facts['stock_flow'] = flows.lower([(spec, laws) for spec, laws, _ in lowered], declarations)


class FlowsProblemPass(Pass):
    """Pass that adds the quantities, flows, and records of the stock and flow structure to the problem."""

    reads = ('problem', 'stock_flow')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Add the stock and flow structure to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            stock_flow=cast('StockFlow', compilation.facts['stock_flow']),
        )


PASSES = (ExpandFlowsPass(), FlowsCensusPass(), LowerFlowsPass(), FlowsProblemPass())
OWNER_RULES = (_flow_owner,)
