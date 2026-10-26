"""Compile the flows of stocks, conserved quantities, and system-dynamics delays of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ....core.base import BaseLaw
from ....core.laws import TransferLaw
from ....core.problems import Problem, StockFlow
from .._compilation import BlockLaws, Compilation
from .._declarations import Declarations
from .._flows import StockFlowPass, quantities_of
from .._pipeline import BasePass
from .._specs import KernelSpec


def _flow_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
    """Return the component that declares a flow, or `None` for another law."""
    del agents
    return spec.owner if isinstance(law, TransferLaw) else None


class ExpandFlowsPass(BasePass):
    """Pass that expands the flows of components into laws of other kinds."""

    reads = ('collected', 'layout')
    writes = ('collected', 'flows')

    def _run(self, compilation: Compilation) -> None:
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


class LowerFlowsPass(BasePass):
    """Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

    reads = ('lowered', 'layout', 'flows')
    writes = ('lowered', 'stock_flow')

    def _run(self, compilation: Compilation) -> None:
        """Lower the flows and their delays.

        Args:
            compilation:
                The compilation state.
        """
        flows = cast('StockFlowPass', compilation.facts['flows'])
        lowered = cast('BlockLaws', compilation.lowered)
        declarations = cast('Declarations', compilation.layout)
        compilation.facts['stock_flow'] = flows.lower([(spec, laws) for spec, laws, _ in lowered], declarations)


class FlowsProblemPass(BasePass):
    """Pass that adds the quantities, flows, and records of the stock and flow structure to the problem."""

    reads = ('problem', 'stock_flow')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the stock and flow structure to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            stock_flow=cast('StockFlow', compilation.facts['stock_flow']),
        )


PASSES = (ExpandFlowsPass(), LowerFlowsPass(), FlowsProblemPass())
OWNER_RULES = (_flow_owner,)
