"""Define the variable spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from sage.all__sagemath_symbolics import RR

from ..types import VariableKind
from ._base_spec import BaseSpec
from ._matrix_group_spec import MatrixGroupSpec


@dataclass(frozen=True)
class VariableSpec(BaseSpec):
    """A spec of a state, parameter, or role of a component, as the compiler reads it.

    It is the declaration as the compiler reads it. A state of order `k` adds the companion states `<name>_d1` to
    `<name>_d<k-1>` that hold its derivatives, and a state with `m` lags adds `<name>_lag1` to `<name>_lag<m>` that hold
    its past values.

    Args:
        kind:
            Whether the argument is a state, a parameter, or a role.

        domain:
            The Sage set of the values, or `None` for a structure rewritten as a whole.

        vector:
            Whether the variable has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give for a state.

        lags:
            The number of past values a state keeps.

        initial:
            The initial values of the derivatives of orders 1 to `order - 1`, or of the lags 1 to `lags`.

        period:
            The period that values of the state wrap around, or `None`.

        derivative_of:
            The state whose derivative a companion state holds, or `None`.

        lag_of:
            The state whose past value a companion state holds, or `None`.

        index:
            The order of the derivative or the lag that a companion state holds.

        variadic:
            Whether a role binds a sequence of entries instead of one path.

        noise:
            Whether a role binds a set of Wiener processes.

        optional:
            Whether a role may be bound to `None`, so it binds nothing.

        per:
            The variadic role whose entries a parameter holds one value for, or `None`.

        quantity:
            The conserved quantity a state holds, or `None`.

        history:
            The values of a state before the start, or `None`.

        adaptive:
            Whether a parameter is adaptive, which laws may change.

        bounds:
            The lower and upper bound of an adaptive parameter, or `None`.

        reference:
            The path of the agent group a member of which a state names by its id, or `None`.

        shape:
            The names of the roles whose group sizes give the shape of a parameter, or `None`.

        tensor:
            The symmetry of a rank-2 tensor state, or `None`.

        algebraic:
            Whether the equations of the system determine the state, which no law integrates.

        group:
            The matrix group whose elements a state takes, or `None`.
    """

    kind: VariableKind
    domain: object = RR
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: tuple[object, ...] = ()
    period: float | None = None
    derivative_of: str | None = None
    lag_of: str | None = None
    index: int = 0
    variadic: bool = False
    noise: bool = False
    optional: bool = False
    per: str | None = None
    quantity: str | None = None
    history: object = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None
    reference: str | None = None
    role: str | None = None
    shape: tuple[str, ...] | None = None
    tensor: str | None = None
    algebraic: bool = False
    group: MatrixGroupSpec | None = None

    @property
    def is_place(self) -> bool:
        """Whether a vector state is the place of its element in a continuous space.

        It is decided by its role.
        """
        return self.vector and self.role in {'position', 'site'}

    @property
    def is_fractional(self) -> bool:
        """Whether the order of the state is not an integer.

        The rate laws of such a state give a Caputo derivative.
        """
        return not float(self.order).is_integer()

    @property
    def derivatives(self) -> int:
        """Number of companion states that hold the derivatives of orders 1 to `order - 1`."""
        return 0 if self.is_fractional else int(self.order) - 1
