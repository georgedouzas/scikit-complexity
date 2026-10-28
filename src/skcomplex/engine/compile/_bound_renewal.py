"""RenewalLaw of a component with its label, the group it renews, and the hidden state that counts it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..problems import Entry
from ._kernel_spec import KernelSpec
from ._renewal_law import RenewalLaw


@dataclass(frozen=True)
class BoundRenewal:
    """RenewalLaw of a component with its label, the group it renews, and the hidden state that counts it.

    Args:
        spec:
            The kernel spec of the component.

        law:
            The renewal law.

        label:
            The label of the renewal.

        group:
            The path of the element or group the renewal resets.

        entry:
            The hidden state that counts the renewals of every member.

        symbol:
            The symbol of the count.
    """

    spec: KernelSpec
    law: RenewalLaw
    label: str
    group: str
    entry: Entry
    symbol: Expression
