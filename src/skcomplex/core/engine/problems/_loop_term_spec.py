"""Define the loop term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class LoopTermSpec(BaseSpec):
    """A spec of one node of the feedback graph of a system: a flow or a rate.

    The feedback graph shows which values drive which, so loops that reinforce or balance themselves can be found, such
    as a population whose growth rate reads its own size. Every node is a flow between stocks or a rate law outside
    flows, with the expression it computes, and the stocks a flow drains and fills. It finds the loops of a population,
    an economy, and a climate model.

    Read more in the user guide, under Analyse.

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

    See Also:
        TransferNetworkSpec: Holds the terms of the feedback graph.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> term = LogisticGrowthSystem().prepare().simulate(until=0).problem.transfer_network.terms[0]
        >>> term.kind, term.path
        ('rate', 'population.x')
    """

    kind: str
    path: str
    expression: Expression
    references: Mapping[Expression, ReferenceSpec]
    source: str | None = None
    sink: str | None = None
