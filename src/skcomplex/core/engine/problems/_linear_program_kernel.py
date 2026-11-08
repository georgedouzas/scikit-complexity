"""Solve the linear program of a problem with HiGHS.

The program is warm-started and solved once per evaluation and per member.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import highspy
import numpy as np
from scipy import sparse

from ...errors import InfeasibleError, SolverError
from ...specs import RunStateSpec
from ...types import Array, Indices
from ..types import LinearProgramBound
from ._base import BaseKernel
from ._linear_program_spec import LinearProgramDualSpec, LinearProgramSpec

INFINITY = highspy.kHighsInf
OBJECTIVE_TOLERANCE = 5e-10
SOLVER_TOLERANCE = 1e-9
ACTIVE_THRESHOLD = 0.0
COLUMN_KINDS = (LinearProgramBound.COLUMN_LOWER, LinearProgramBound.COLUMN_UPPER)
INFEASIBLE_STATUSES = (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnboundedOrInfeasible)


class LinearProgramKernel(BaseKernel):
    """Linear program lowered for a problem.

    It is solved with one warm-started HiGHS system per member. `prepare(state)` evaluates the bounds and coefficients
    at a state without solving. `run(state)` keeps in `values_` a copy of the state vector with the variables, objective
    values, and duals of every member set. The solve of a time and of the values the program reads runs once, so another
    run at them only copies its values.

    Args:
        program:
            The lowered program.

    Attributes:
        matrix_:
            The coefficients of the rows at the last prepared or solved state, of the first active member or else the
            first member.

        lower_:
            The lower bound of every variable of every member at the last prepared or solved state.

        upper_:
            The upper bound of every variable of every member at the last prepared or solved state.

        row_lower_:
            The lower bound of every row of every member at the last prepared or solved state.

        row_upper_:
            The upper bound of every row of every member at the last prepared or solved state.

        solves_:
            The number of solves so far, one per member that solved at a time and at new values.

        values_:
            The state vector with the values of the program of the last run.

    Raises:
        InfeasibleError:
            A program with the policy `'raise'` has no solution, raised by `run`.

        SolverError:
            A program is unbounded, or HiGHS fails, raised by `run`.

    Examples:
        >>> import numpy as np
        >>> from scipy import sparse
        >>> from skcomplex.core.engine.problems._linear_program_kernel import LinearProgramKernel
        >>> from skcomplex.core.engine.problems._linear_program_spec import LinearProgramSpec
        >>> from skcomplex.core.specs import RunStateSpec
        >>> program = LinearProgramSpec(
        ...     path='flows',
        ...     names=('flows.v',),
        ...     variables=(np.array([0]),),
        ...     constraints=(),
        ...     relations=('v <= 3',),
        ...     matrix=sparse.csr_array((0, 1)),
        ...     lower=np.array([0.0]),
        ...     upper=np.array([3.0]),
        ...     row_lower=np.zeros(0),
        ...     row_upper=np.zeros(0),
        ...     costs=np.array([[-1.0]]),
        ...     signs=(-1.0,),
        ...     offsets=(0.0,),
        ...     objectives=(np.array([1]),),
        ... )
        >>> kernel = LinearProgramKernel(program).prepare()
        >>> kernel.run(RunStateSpec(0.0, np.zeros(2), np.zeros(0))).values_.tolist()
        [3.0, 3.0]
    """

    values_: Array

    def __init__(self, program: LinearProgramSpec) -> None:
        self.program = program

    def _prepare(self, context: object | None) -> None:
        """Forget the systems and the last solve.

        When a state is given, the bounds and coefficients are evaluated at it.

        Args:
            context:
                The state to evaluate at, a `RunStateSpec`, or `None`.
        """
        program = self.program
        places = [*program.variables, *program.objectives, *(item.indices for item in program.duals)]
        self._places = np.concatenate(places).astype(np.intp)
        self._side_rows = sorted({item.position for item in program.sides if item.kind not in COLUMN_KINDS})
        self.solves_ = 0
        self._systems: dict[int, highspy.Highs] = {}
        self._bounds: dict[int, tuple[Array, Array]] = {}
        self._key: tuple[float, bytes] | None = None
        self._last: Array | None = None
        self._first_solution: highspy.HighsSolution | None = None
        if isinstance(context, RunStateSpec):
            self._set_sides(self._compiled_values(context))

    def _compiled_values(self, state: RunStateSpec) -> list[Array]:
        """Return the compiled values at a state, one array with a value per member each."""
        program = self.program
        if program.function is None:
            return []
        columns = [reference.column_of(state.x, state.p) for reference in program.arguments]
        values = program.function(state.t, *columns)
        return [np.broadcast_to(np.asarray(item, dtype=float), (program.members,)) for item in values]

    def _set_sides(self, values: Sequence[Array]) -> None:
        """Set the bounds and coefficients of every member from the compiled values."""
        program = self.program
        rows = program.members
        lower, upper = np.tile(program.lower, (rows, 1)), np.tile(program.upper, (rows, 1))
        row_lower, row_upper = np.tile(program.row_lower, (rows, 1)), np.tile(program.row_upper, (rows, 1))
        for side in program.sides:
            value = values[side.value] / side.scale
            if side.kind == LinearProgramBound.COLUMN_LOWER:
                lower[:, side.position] = np.maximum(lower[:, side.position], value)
            elif side.kind == LinearProgramBound.COLUMN_UPPER:
                upper[:, side.position] = np.minimum(upper[:, side.position], value)
            elif side.kind == LinearProgramBound.ROW_LOWER:
                row_lower[:, side.position] = value
            else:
                row_upper[:, side.position] = value
        self.lower_, self.upper_, self.row_lower_, self.row_upper_ = lower, upper, row_lower, row_upper
        self.matrix_ = self._matrix_of(0, values)

    def _matrix_of(self, member: int, values: Sequence[Array]) -> sparse.csr_array:
        """Return the coefficients of the rows of a member."""
        program = self.program
        if not program.entries:
            return program.matrix
        matrix = program.matrix.tolil()
        for entry in program.entries:
            matrix[entry.row, entry.column] = values[entry.value][member]
        return matrix.tocsr()

    def _new_system(self) -> highspy.Highs:
        """Return a HiGHS system of the program.

        The system holds its parts of absolute values and its rows that fix objectives.
        """
        program = self.program
        size, count = len(program.names), len(program.parts)
        extended = sparse.hstack([program.matrix, sparse.csr_array((program.matrix.shape[0], 2 * count))])
        rows = [extended]
        if count:
            own = sparse.csr_array((np.ones(count), (np.arange(count), np.asarray(program.parts))), shape=(count, size))
            rows.append(sparse.hstack([own, -sparse.eye_array(count), sparse.eye_array(count)]))
        if program.costs.shape[0] > 1:
            rows.append(sparse.csr_array(program.costs[:-1]))
        matrix = sparse.vstack(rows).tocsr() if len(rows) > 1 else sparse.csr_array(extended)
        columns, height = matrix.shape[1], matrix.shape[0]
        system = highspy.Highs()
        system.setOptionValue('output_flag', False)
        system.setOptionValue('threads', 1)
        system.setOptionValue('presolve', 'off')
        system.setOptionValue('solver', 'simplex')
        system.setOptionValue('primal_feasibility_tolerance', SOLVER_TOLERANCE)
        system.setOptionValue('dual_feasibility_tolerance', SOLVER_TOLERANCE)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = columns, height
        lp.col_cost_ = np.zeros(columns)
        lp.col_lower_ = np.concatenate([program.lower, np.zeros(2 * count)])
        lp.col_upper_ = np.concatenate([program.upper, np.full(2 * count, INFINITY)])
        fixed = height - len(program.row_lower) - count
        lp.row_lower_ = np.concatenate([program.row_lower, np.zeros(count), np.full(fixed, -INFINITY)])
        lp.row_upper_ = np.concatenate([program.row_upper, np.zeros(count), np.full(fixed, INFINITY)])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = matrix.indptr
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data
        system.passModel(lp)
        return system

    def _system_of(self, member: int) -> highspy.Highs:
        """Return the system of a member.

        The system keeps its basis from one solve to the next.
        """
        system = self._systems.get(member)
        if system is None:
            system = self._systems[member] = self._new_system()
        return system

    def _set_bounds(self, system: highspy.Highs, member: int) -> None:
        """Change only the bounds of a system, and open the rows that fix objectives."""
        program = self.program
        columns = len(program.names)
        lower, upper = self.lower_[member], self.upper_[member]
        last = self._bounds.get(member)
        changed = np.arange(columns) if last is None else np.flatnonzero((last[0] != lower) | (last[1] != upper))
        if changed.size:
            system.changeColsBounds(changed.size, changed, lower[changed], upper[changed])
        self._bounds[member] = (lower, upper)
        for row in self._side_rows:
            system.changeRowBounds(row, self.row_lower_[member, row], self.row_upper_[member, row])
        start = len(program.row_lower) + len(program.parts)
        for row in range(start, start + program.costs.shape[0] - 1):
            system.changeRowBounds(row, -INFINITY, INFINITY)

    def _costs_of(self, member: int, values: Sequence[Array]) -> Array:
        """Return the cost of every column per objective for a member.

        Costs are in the minimising sense.
        """
        program = self.program
        if not program.cost_entries:
            return program.costs
        costs = program.costs.copy()
        for entry in program.cost_entries:
            costs[entry.row, entry.column] = program.signs[entry.row] * values[entry.value][member]
        return costs

    def _set_coefficients(self, system: highspy.Highs, costs: Array, values: Sequence[Array], member: int) -> None:
        """Change the coefficients that compiled values give.

        They change in the matrix and in the rows that fix objectives.
        """
        program = self.program
        for entry in program.entries:
            system.changeCoeff(entry.row, entry.column, float(values[entry.value][member]))
        start = len(program.row_lower) + len(program.parts)
        for entry in program.cost_entries:
            if entry.row < costs.shape[0] - 1:
                system.changeCoeff(start + entry.row, entry.column, float(costs[entry.row, entry.column]))

    def _describe_subset(self, system: highspy.Highs) -> str:
        """Return the constraints and variables of an irreducible infeasible subset.

        The subset is one of a system.
        """
        program = self.program
        _, subset = system.getIis()
        rows = [program.constraints[item] for item in subset.row_index_ if item < len(program.constraints)]
        columns = [program.names[item] for item in subset.col_index_ if item < len(program.names)]
        return f'the constraints {"; ".join(rows) or "none"} and the variables {", ".join(columns) or "none"}'

    def _raise_infeasible(self, system: highspy.Highs, t: float, member: int) -> None:
        """Raise the error of an infeasible solve.

        The error names the program, the time, the member, and a subset.
        """
        program = self.program
        place = f' for member {member}' if program.members > 1 else ''
        error_msg = (
            f'The linear program of {program.path} has no solution at time {t}{place}: '
            f'{self._describe_subset(system)} cannot hold together. Relax one of them, or give the law the policy '
            "`infeasible='zero'` or `infeasible='hold'`."
        )
        raise InfeasibleError(error_msg)

    @staticmethod
    def _solved_status(system: highspy.Highs) -> highspy.HighsModelStatus:
        """Run a system, retrying when a warm start ends without an answer.

        The retry runs it from no basis, then also with presolve.
        """
        system.run()
        status = system.getModelStatus()
        for presolve in ('off', 'on'):
            if status == highspy.HighsModelStatus.kOptimal or status in INFEASIBLE_STATUSES:
                break
            system.clearSolver()
            system.setOptionValue('presolve', presolve)
            system.run()
            status = system.getModelStatus()
        system.setOptionValue('presolve', 'off')
        return status

    def _objective_values(self, system: highspy.Highs, costs: Array) -> list[float] | None:
        """Solve the objectives in order and return their values.

        Each objective is kept near its optimum. Without objectives, `None` is returned.
        """
        program = self.program
        columns = len(program.names) + 2 * len(program.parts)
        positions = np.arange(columns)
        found: list[float] = []
        start = len(program.row_lower) + len(program.parts)
        for index in range(max(1, costs.shape[0])):
            system.changeColsCost(columns, positions, costs[index] if costs.size else np.zeros(columns))
            status = self._solved_status(system)
            if status in INFEASIBLE_STATUSES:
                return None
            if status != highspy.HighsModelStatus.kOptimal:
                error_msg = (
                    f'The linear program of {program.path} did not reach an optimum: HiGHS reports '
                    f'{system.modelStatusToString(status)}. Bound the variables its objective reads.'
                )
                raise SolverError(error_msg)
            if index == 0:
                self._first_solution = system.getSolution()
            if not costs.size:
                return found
            value = float(system.getInfo().objective_function_value)
            found.append(value)
            if index < costs.shape[0] - 1:
                slack = OBJECTIVE_TOLERANCE * min(1.0, abs(value))
                system.changeRowBounds(start + index, -INFINITY, value + slack)
        return found

    def _dual_values(self, member: int, solution: highspy.HighsSolution | None) -> list[float]:
        """Return the dual of every recorded constraint.

        It is the change of the first objective with its right side.
        """
        program = self.program
        sign = program.signs[0] if program.signs else 1.0
        if solution is None:
            return [0.0] * len(program.duals)
        return [self._dual_of(item, member, solution, sign) for item in program.duals]

    def _dual_of(self, item: LinearProgramDualSpec, member: int, solution: highspy.HighsSolution, sign: float) -> float:
        """Return the dual of one constraint.

        It is a row dual, or the reduced cost of a variable at its bound.
        """
        if item.row >= 0:
            return sign * float(solution.row_dual[item.row])
        value = float(solution.col_value[item.column])
        sides = [
            (
                self.lower_[member, item.column]
                if kind == LinearProgramBound.COLUMN_LOWER
                else self.upper_[member, item.column]
            )
            for kind in item.kinds
        ]
        if not any(abs(value - side) <= SOLVER_TOLERANCE * max(1.0, abs(value)) for side in sides):
            return 0.0
        return sign * float(solution.col_dual[item.column]) * item.scale

    def _solve_member(self, member: int, values: Sequence[Array], t: float) -> Array | None:
        """Return the values of a member after its solve.

        An infeasible solve returns `None` for the policy to decide.
        """
        program = self.program
        system = self._system_of(member)
        self._set_bounds(system, member)
        costs = self._costs_of(member, values)
        if program.entries or program.cost_entries:
            self._set_coefficients(system, costs, values, member)
        self.solves_ += 1
        objectives = self._objective_values(system, costs)
        if objectives is None:
            if program.infeasible == 'raise':
                self._raise_infeasible(system, t, member)
            return None
        size = len(program.names)
        solution = system.getSolution()
        variables = np.asarray(solution.col_value, dtype=float)[:size]
        shown = [
            sign * value + offset
            for sign, value, offset in zip(program.signs, objectives, program.offsets, strict=True)
        ]
        duals = self._dual_values(member, self._first_solution) if program.duals else []
        return np.concatenate([variables, shown, duals])

    def _active_members(self, values: Sequence[Array]) -> Indices:
        """Return the members whose activity holds."""
        program = self.program
        holds = np.ones(program.members, dtype=bool)
        for position in program.activity:
            holds &= values[position] > ACTIVE_THRESHOLD
        return np.flatnonzero(holds)

    def _solved_values(self, state: RunStateSpec, last: Array) -> Array:
        """Return the values of every member at a state, one row per member."""
        values = self._compiled_values(state)
        self._set_sides(values)
        following = np.zeros_like(last)
        for member in self._active_members(values):
            row = self._solve_member(int(member), values, state.t)
            if row is not None:
                following[member] = row
            elif self.program.infeasible == 'hold':
                following[member] = last[member]
        return following

    def _key_of(self, state: RunStateSpec) -> tuple[float, bytes]:
        """Return what decides a solve.

        It is the time and the values the program reads.
        """
        columns = [reference.column_of(state.x, state.p) for reference in self.program.arguments]
        return state.t, b''.join(item.tobytes() for item in columns)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Solve the program at a state and set `values_`.

        The solve is skipped when the last solve read the same values.

        Args:
            inputs:
                The state.
        """
        state = self._state_of(inputs)
        places = self._places
        members = self.program.members
        key = self._key_of(state)
        if self._last is None:
            self._last = state.x[places].reshape(members, -1, order='F').copy()
        if key != self._key:
            self._last = self._solved_values(state, self._last)
            self._key = key
        following = state.x.copy()
        following[places] = self._last.ravel(order='F')
        self.values_ = following
