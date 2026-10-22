"""Define expectation observables of operators on quantum systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any

from ..components._quantum import _members
from ._observables import Observable


def expectation(
    label: str,
    expression: Callable[..., Any],
    *,
    group: str | None = None,
    reduce: str | Callable[[Any], Any] | None = None,
) -> Observable:
    """Return the observable of the expectation value of an operator.

    For a system, `expression(s)` returns an operator of the system namespace, where `s.<element>` is the namespace of
    a quantum element and `s.<group>` is the tuple of the namespaces of its members. For a group, `expression(m)`
    returns an operator of one member, and the observable holds one value per member unless it is reduced. The value is
    `<psi|O|psi>` for a pure state and `Tr(rho O)` for a density matrix, real for a Hermitian operator.

    Args:
        label:
            The label of the observable.

        expression:
            A module-level function of the namespace that returns an operator.

        group:
            The path of the quantum group to evaluate member by member, or `None` for the whole system.

        reduce:
            The reduction of the member values, such as `'sum'` or `'mean'`, or `None` to keep one value per member.

    Returns:
        observable:
            The observable.

    Examples:
        >>> from skcomplex.core.system import expectation
        >>> expectation('sz', abs, group='chain').group
        'chain'
    """
    return Observable(label, expression, group=group, reduce=_members if reduce is None else reduce)
