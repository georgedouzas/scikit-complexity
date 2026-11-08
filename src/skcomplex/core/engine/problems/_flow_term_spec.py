"""Define the flow term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _FlowTermSpec(BaseSpec):
    """Flow or rate of a spec, with its source and sink.

    It becomes a node of the feedback graph.

    Args:
        spec:
            The spec whose references the expression reads.

        kind:
            `'flow'`, or `'rate'` for a rate law outside flows.

        path:
            The node of a flow, or the state a rate law targets.

        expression:
            The expression.

        source:
            The state a flow drains, or `None`.

        sink:
            The state a flow fills, or `None`.
    """

    spec: KernelSpec
    kind: str
    path: str
    expression: Expression
    source: str | None = None
    sink: str | None = None
