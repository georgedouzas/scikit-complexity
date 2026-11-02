"""Helpers shared by the synchronization models: the order parameter of a group of phases."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, I, exp

from ....specs import ObservableSpec


def _phase_vector(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's phase in the complex plane."""
    return exp(I * member.theta)


def _modulus(name: str, s: SimpleNamespace) -> Expression:
    """Return the modulus of an observable of a system."""
    return abs(getattr(s, name))


def _order_parameter(label: str, group: str) -> list[ObservableSpec]:
    """Return the complex mean field of a group's phases and its modulus, the order parameter."""
    return [
        ObservableSpec(f'{label}_field', _phase_vector, group=group),
        ObservableSpec(label, partial(_modulus, f'{label}_field')),
    ]
