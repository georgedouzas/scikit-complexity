"""Define the order parameter of a group of phases.

The order parameter is the modulus of their complex mean field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, I, exp

from .....core.specs import ObservableSpec


def _phase_vector(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's phase in the complex plane."""
    return exp(I * member.theta)


def _modulus(name: str, s: SimpleNamespace) -> Expression:
    """Return the modulus of an observable of a system."""
    return abs(getattr(s, name))


def order_parameter(label: str, group: str) -> list[ObservableSpec]:
    """Return the complex mean field of a group's phases and its modulus.

    The modulus is the order parameter.

    The mean field `Z = <exp(i theta)>` averages the unit vectors of the phases `theta` of the group's members, and the
    order parameter `r = |Z|` is 0 for scattered phases and 1 for locked phases.

    Args:
        label:
            The label of the order parameter. The mean field has the label `<label>_field`.

        group:
            The path of the group of oscillators, whose members have the phase `theta`.

    Returns:
        observables:
            The mean field and the order parameter.

    Examples:
        >>> from skcomplex.systems.physics.common.oscillators import order_parameter
        >>> [observable.label for observable in order_parameter('order', 'oscillators')]
        ['order_field', 'order']
    """
    return [
        ObservableSpec(f'{label}_field', _phase_vector, group=group),
        ObservableSpec(label, partial(_modulus, f'{label}_field')),
    ]
