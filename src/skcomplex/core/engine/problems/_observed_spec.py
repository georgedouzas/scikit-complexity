"""Define the observed spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class ObservedSpec(BaseSpec):
    """A spec of an observable that a law reads through a role, at the current state.

    A law may read an observable of another system, such as the total population of a region that sets a price, through
    a role bound to its path. The spec holds the namespace the observable is computed in and the value every symbol of
    it stands for, so the engine computes the observable at the state the law runs at. It feeds the total population to
    a market, the mean temperature to a controller, and the market price to a firm.

    Read more in the user guide, under Observe.

    Args:
        namespace:
            The namespace the expression of the observable reads.

        references:
            The value every symbol of the namespace stands for.

    See Also:
        Observable: The observable a role reads.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.engine.problems._observed_spec import ObservedSpec
        >>> ObservedSpec(SimpleNamespace(), {}).references
        {}
    """

    namespace: SimpleNamespace
    references: dict[Expression, ReferenceSpec]
