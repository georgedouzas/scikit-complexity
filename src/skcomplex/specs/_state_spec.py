"""Define the state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import RR

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._tensor_spec import TENSOR_SYMMETRIES

ROLES = ('heading', 'velocity', None)


@dataclass(frozen=True)
class StateSpec(BaseSpec):
    """Spec of a state of an element, in the annotation of an argument of its constructor.

    The default of the argument is the initial value of the state. A state of order `k` has its `k`-th time derivative
    given by its rate laws, and a state with `m` lags keeps its past values for difference equations.

    Args:
        domain:
            The Sage set of the values of the state, such as `RR`, `ZZ`, `RealSet.closed_open(0, oo)`, or
            `FiniteEnumeratedSet(['S', 'I', 'R'])`, or `None` for a structure rewritten as a whole, such as a word,
            whose element checks it. A state of an agent with the domain `None` or `object` holds any hashable value,
            such as a tuple, a string, or a frozen set.

        vector:
            Whether the state has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give, an integer of at least 1, or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order.

        lags:
            The number of past values the state keeps for difference equations, 0 for none.

        initial:
            The initial values of the derivatives of orders 1 to `order - 1`, or of the past values at steps -1 to
            `-lags`.

        period:
            The period that values of the state wrap around, such as `2 * pi` for an angle, or `None` for none.

        quantity:
            The conserved quantity the state holds, such as `'money'`, or `None`. Only transfers change a tagged state.

        history:
            The values of the state before the start, which delayed laws read, a number or a function of time, or
            `None` to keep the initial value before the start.

        position:
            Whether a vector state is the position of its element in a continuous space, on which the boundary of the
            space acts.

        reference:
            The path of the agent group, relative to the element's system, a member of which the state names by its
            id, or `None`. A negative value names no member, nor does any value in a model without that group. The
            initial values of the group give the position of the member among the initial members of the referenced
            group, which the agent stepper turns into its id, and rules read the member with `agent.referenced(state)`.

        role:
            What the state means in a snapshot of agents, `'heading'` for an angle or `'velocity'` for one component of
            a velocity, or `None` for any other state. Analyses read headings and velocities from these roles alone, so
            a state named `phase` is no heading unless it says so.

        tensor:
            The symmetry of a rank-2 tensor state, `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which
            means symmetric and traceless, or `None` for a state that is no tensor. The state stores its independent
            components, and laws read it as a Sage matrix whose dependent entries are written in terms of them.

    Examples:
        >>> from sage.all__sagemath_symbolics import oo
        >>> from sage.sets.real_set import RealSet
        >>> from skcomplex.specs import StateSpec
        >>> spec = StateSpec(domain=RealSet.closed_open(0, oo)).check()
        >>> 2.5 in spec.domain, -1 in spec.domain
        (True, False)
        >>> StateSpec(reference='banks').check().reference
        'banks'
        >>> StateSpec(tensor='traceless').check().tensor
        'traceless'
    """

    domain: object = RR
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: Sequence[object] = ()
    period: float | None = None
    quantity: str | None = None
    history: object = None
    position: bool = False
    reference: str | None = None
    role: str | None = None
    tensor: str | None = None

    def _check(self) -> None:
        """Raise an error for an invalid order, lags, or initial values, or a reference of a state that is not scalar.

        Raises:
            ValidationError:
                The order or the lags are invalid, there are more initial values than derivatives or lags, or a state
                naming a member of a group is not a scalar of order 1 without lags, or the tensor symmetry is unknown or
                given together with a vector.
        """
        if self.tensor is not None and self.vector:
            error_msg = f'A state takes either vector=True or tensor={self.tensor!r}, not both.'
            raise ValidationError(error_msg)
        if self.tensor is not None and self.tensor not in TENSOR_SYMMETRIES:
            error_msg = f'The tensor symmetry is {self.tensor!r}. Use one of {", ".join(TENSOR_SYMMETRIES)}.'
            raise ValidationError(error_msg)
        if self.role not in ROLES:
            error_msg = f"The role of a state is 'heading', 'velocity', or None, but got {self.role!r}."
            raise ValidationError(error_msg)
        if self.reference is not None and (self.vector or self.tensor or self.order != 1 or self.lags):
            error_msg = f'The state naming a member of {self.reference!r} is not a scalar of order 1 without lags.'
            raise ValidationError(error_msg)
        integral = float(self.order).is_integer()
        if not (math.isfinite(self.order) and self.order > 0 and (integral or self.order < 1)):
            error_msg = f'The order {self.order!r} is neither an integer of at least 1 nor a real number in (0, 1).'
            raise ValidationError(error_msg)
        if not (isinstance(self.lags, int) and self.lags >= 0):
            error_msg = f'The lags {self.lags!r} are not a nonnegative integer.'
            raise ValidationError(error_msg)
        if self.order != 1 and self.lags:
            error_msg = 'A state takes either an order other than 1 or lags, not both.'
            raise ValidationError(error_msg)
        companions = max(int(self.order) - 1 if integral else 0, self.lags)
        if len(self.initial) > companions:
            error_msg = f'{len(self.initial)} initial values were given for {companions} derivatives or lags.'
            raise ValidationError(error_msg)
