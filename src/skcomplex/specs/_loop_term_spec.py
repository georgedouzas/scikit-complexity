"""Define the loop term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class LoopTermSpec(BaseSpec):
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
