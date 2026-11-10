"""Define the spec of a linear program lowered from laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ...specs._base_spec import BaseSpec
from ...types import Array, Indices
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class LinearProgramSideSpec(BaseSpec):
    """A spec of a bound of a column or a row that a compiled value gives.

    Args:
        kind:
            The bound it sets, one of `COLUMN_LOWER`, `COLUMN_UPPER`, `ROW_LOWER`, and `ROW_UPPER`.

        position:
            The column or the row.

        value:
            The position of the compiled value.

        scale:
            The number the value is divided by.
    """

    kind: int
    position: int
    value: int
    scale: float = 1.0


@dataclass(frozen=True)
class LinearProgramEntrySpec(BaseSpec):
    """A spec of a coefficient of the matrix or of an objective of a linear program.

    A compiled value gives it at every solve.

    Args:
        row:
            The row of the matrix, or the position of the objective.

        column:
            The column.

        value:
            The position of the compiled value.
    """

    row: int
    column: int
    value: int


@dataclass(frozen=True)
class LinearProgramDualSpec(BaseSpec):
    """A spec of where the dual of one constraint of a law is recorded.

    Args:
        indices:
            The flat indices of the dual of every member.

        constraint:
            The text of the constraint.

        row:
            The row of the constraint, or `-1` for a bound.

        column:
            The column of a bound, or `-1` for a row.

        kinds:
            The bounds of the column that the relation gives, `COLUMN_LOWER`, `COLUMN_UPPER`, or both.

        scale:
            The coefficient of the variable of a bound, which turns its reduced cost into the dual of the relation.
    """

    indices: Indices = field(repr=False)
    constraint: str
    row: int = -1
    column: int = -1
    kinds: tuple[int, ...] = ()
    scale: float = 1.0


@dataclass(frozen=True)
class LinearProgramSpec(BaseSpec):
    """A spec of a linear program lowered from the laws that share its variables.

    Args:
        path:
            The path of the element or group that holds its objectives, or of its first law.

        names:
            The dotted name of every variable.

        variables:
            The flat indices of every variable of every member, one array per variable.

        constraints:
            The text of every row of the matrix.

        relations:
            The text of every relation of the laws, with the rows of the matrix and the bounds of the variables.

        matrix:
            The coefficients of the rows, with `0` for the coefficients that `entries` gives.

        lower:
            The constant lower bound of every variable.

        upper:
            The constant upper bound of every variable.

        row_lower:
            The constant lower bound of every row.

        row_upper:
            The constant upper bound of every row.

        sides:
            The bounds that compiled values give.

        entries:
            The coefficients of the matrix that compiled values give.

        function:
            The compiled function `f(t, *arguments)` of the values, or `None` for a program without any.

        arguments:
            The references that the compiled function reads.

        members:
            The number of members that solve the program.

        activity:
            The positions of the compiled values that must be positive for a member to solve.

        costs:
            The cost of every variable and of the positive and negative parts of its absolute value, one row per
            objective, in the minimising sense.

        cost_entries:
            The costs that compiled values give, in the sense of the objective.

        signs:
            `1` for an objective that is minimised and `-1` for one that is maximised.

        offsets:
            The constant term of every objective.

        parts:
            The variables whose absolute values the objectives read, in the order of their parts.

        objectives:
            The flat indices of the value of every objective of every member, one array per objective.

        duals:
            The places of the duals to record.

        infeasible:
            The infeasibility policy.
    """

    path: str
    names: tuple[str, ...]
    variables: tuple[Indices, ...] = field(repr=False)
    constraints: tuple[str, ...] = field(repr=False)
    relations: tuple[str, ...] = field(repr=False)
    matrix: sparse.csr_array = field(repr=False)
    lower: Array = field(repr=False)
    upper: Array = field(repr=False)
    row_lower: Array = field(repr=False)
    row_upper: Array = field(repr=False)
    sides: tuple[LinearProgramSideSpec, ...] = field(default=(), repr=False)
    entries: tuple[LinearProgramEntrySpec, ...] = field(default=(), repr=False)
    function: Callable[..., list[npt.ArrayLike]] | None = field(default=None, repr=False)
    arguments: tuple[ReferenceSpec, ...] = field(default=(), repr=False)
    members: int = 1
    activity: tuple[int, ...] = ()
    costs: Array = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    cost_entries: tuple[LinearProgramEntrySpec, ...] = field(default=(), repr=False)
    signs: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    parts: tuple[int, ...] = ()
    objectives: tuple[Indices, ...] = field(default=(), repr=False)
    duals: tuple[LinearProgramDualSpec, ...] = field(default=(), repr=False)
    infeasible: str = 'raise'
