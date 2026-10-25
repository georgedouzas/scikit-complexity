"""Helpers shared by the synchronization models: the order parameter of a group of phases."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

from sage.all__sagemath_symbolics import I, exp

from ....core.base import Expression
from ....core.laws import Symbols
from ....core.system import ObservableQuantity


def _phase_vector(member: Symbols) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's phase in the complex plane."""
    return exp(I * member.theta)


def _modulus(name: str, s: Symbols) -> Expression:
    """Return the modulus of an observable of a system."""
    return abs(getattr(s, name))


def _order_parameter(label: str, group: str) -> list[ObservableQuantity]:
    """Return the complex mean field of a group's phases and its modulus, the order parameter."""
    return [
        ObservableQuantity(f'{label}_field', _phase_vector, group=group),
        ObservableQuantity(label, partial(_modulus, f'{label}_field')),
    ]
