"""Define the spec of a linear program lowered from laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ...base import BaseSpec
from ...types import Array, Indices
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class LinearProgramSideSpec(BaseSpec):
    """A spec of a bound of a linear program that a value computed during a run sets.

    A bound of a variable or a constraint may depend on the states or parameters, such as the uptake limit of a nutrient
    that falls as the medium runs out, so it is computed again before every solve. The spec says which bound the value
    sets, where, and what it is divided by. It sets the uptake limits of a cell, the capacity of a pipe, and the budget
    of a firm.

    Read more in the user guide, under Solvers.

    Args:
        kind:
            The bound it sets, one of `COLUMN_LOWER`, `COLUMN_UPPER`, `ROW_LOWER`, and `ROW_UPPER`.

        position:
            The column or the row.

        value:
            The position of the compiled value.

        scale:
            The number the value is divided by.

    See Also:
        LinearProgramBound: The four kinds of bound.
        LinearProgramSpec: The program that holds the bounds.

    Examples:
        >>> from skcomplex.core.engine.enums import LinearProgramBound
        >>> from skcomplex.core.engine.problems._linear_program_spec import LinearProgramSideSpec
        >>> LinearProgramSideSpec(LinearProgramBound.COLUMN_UPPER, 0, 1).scale
        1.0
    """

    kind: int
    position: int
    value: int
    scale: float = 1.0


@dataclass(frozen=True)
class LinearProgramEntrySpec(BaseSpec):
    """A spec of a coefficient of a linear program that a run computes again.

    A coefficient of a constraint or of an objective may depend on the states or parameters, such as the yield of a
    reaction that falls with temperature, so it is computed again before every solve. The spec says which row and column
    it sets and where its value sits. It sets the yields of a cell, the losses of a pipe, and the prices of a firm.

    Read more in the user guide, under Solvers.

    Args:
        row:
            The row of the matrix, or the position of the objective.

        column:
            The column.

        value:
            The position of the compiled value.

    See Also:
        LinearProgramSpec: The program that holds the coefficients.

    Examples:
        >>> from skcomplex.core.engine.problems._linear_program_spec import LinearProgramEntrySpec
        >>> LinearProgramEntrySpec(row=0, column=2, value=1).column
        2
    """

    row: int
    column: int
    value: int


@dataclass(frozen=True)
class LinearProgramDualSpec(BaseSpec):
    """A spec of where the dual value of a constraint of a linear program is kept.

    The dual value of a constraint says how much the best objective would improve if the constraint were loosened by one
    unit, such as the worth of one more unit of a nutrient to the growth of a cell. The spec says which row or bound
    gives it and where it is written in the state of every member. It prices the nutrients of a cell, the capacity of a
    pipe, and the budget of a firm.

    Read more in the user guide, under Solvers.

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

    See Also:
        LinearProgramSpec: The program that records the duals.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems._linear_program_spec import LinearProgramDualSpec
        >>> LinearProgramDualSpec(np.array([3]), 'v <= 3').row
        -1
    """

    indices: Indices = field(repr=False)
    constraint: str
    row: int = -1
    column: int = -1
    kinds: tuple[int, ...] = ()
    scale: float = 1.0


@dataclass(frozen=True)
class LinearProgramSpec(BaseSpec):
    """A spec of a linear program built from the laws that share its variables.

    The objective and constraint laws of a system that share variables, such as the fluxes of a cell, form one linear
    program: a matrix of constraints, bounds on the variables and constraints, the costs of the objectives, and the
    coefficients and bounds that are computed again before every solve. The kernel of the program solves it for every
    member. It holds the metabolism of a cell, the flows of a supply chain, and the dispatch of a power grid.

    Read more in the user guide, under Solvers.

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

    See Also:
        LinearProgramKernel: Solves the program.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FluxBalanceSystem
        >>> program = FluxBalanceSystem().prepare().simulate(until=0).problem.programs[0].program
        >>> program.path, program.members
        ('cell', 1)
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
