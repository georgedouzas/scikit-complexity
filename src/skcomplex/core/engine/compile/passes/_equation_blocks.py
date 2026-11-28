"""Match, order, and solve the equations of algebraic states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms import bipartite
from sage.all__sagemath_symbolics import SR, Expression, matrix
from sage.modules.free_module_element import FreeModuleElement
from scipy.optimize import root

from ....components.laws import BaseLaw, CallSpec, EventLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws._boundary import _BoundaryLaw
from ....errors import SolverError, ValidationError
from ....types import Array
from ..._utils import symbols_of, vector
from ...problems._expression_printer import ExpressionPrinter
from ...problems._kernel_spec import KernelSpec
from ...problems._reference_spec import ReferenceSpec
from .._algebraic_law import AlgebraicLaw
from .._base import BaseLowering, BasePass
from .._compilation import Compilation, Laws
from .._issue_spec import IssueSpec
from .._laws import KERNEL_FREE_LAWS
from .._renewal_law import RenewalLaw

SYMBOLIC_UNKNOWNS = 6
ROOT_TOLERANCE = 1e-12
RESIDUAL_TOLERANCE = 1e-8


type _Printed = Callable[..., list[npt.ArrayLike]]
type _BlockSolution = Callable[[float, list[float]], list[float]]


@dataclass(frozen=True)
class _Relation:
    """Equation of a component as `expression == 0`, with its unknowns.

    It holds the states the unknowns stand for, by their symbols. The unknowns are the placeholders of its highest
    derivatives and the symbols of its algebraic states.

    The key names the state whose equation it is, as its entry and column, so that the relations of one key add, or is
    `None` for a relation of its own. The target is the symbol of that state.
    """

    spec: KernelSpec
    expression: Expression
    highest: dict[Expression, Expression]
    description: str
    key: tuple[str, int] | None = None
    target: Expression | None = None


@dataclass(frozen=True)
class _NumericBlock:
    """Equations `residuals == 0` that a block solves numerically for its unknowns.

    The block solves them at every evaluation.
    """

    unknowns: tuple[Expression, ...]
    residuals: tuple[Expression, ...]
    linear: bool


class EquationBlocksLowering(BaseLowering[Laws, Sequence[tuple[KernelSpec, list[BaseLaw | _Relation]]]]):
    """A lowering that solves the equations of a system for its algebraic states.

    An algebraic state is fixed at every moment by equations, such as the currents of a circuit that Kirchhoff's laws
    fix. `run(specs)` matches every equation to the unknown it solves, orders the matched pairs into blocks that can be
    solved one after the other, and solves every block: a small linear block by hand, a large linear block by a linear
    solve, and any other block numerically by `scipy.optimize.root`. The laws that read algebraic states then read their
    solutions, and `lowered_` holds a rate law per highest derivative and an algebraic law per algebraic state. It
    solves circuits, motors, and networks of pipes.

    Read more in the user guide, under Extend the library.

    Args:
        relations:
            The relations of the system, with those of one target added into one.

        unknowns:
            The state each unknown stands for, by the placeholder of a highest derivative or the symbol of an algebraic
            state.

    Attributes:
        guesses_:
            The starting value of every algebraic unknown, by its symbol.

        numeric_:
            The unknowns the numeric blocks determine.

        algebraic_:
            The solution of every algebraic state, which holds the numeric unknowns of a numeric block.

        arguments_:
            The symbols the numeric blocks read.

        solve_:
            The function of time and `arguments_` that returns the values of `numeric_`.

    See Also:
        AlgebraicLaw: Gives an algebraic state its value.
        OrdersPass: The pass that runs this lowering.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.engineering.mechatronics import DcMotorSystem
        >>> compilation = PassCompiler().prepare(DcMotorSystem()).compilation_
        >>> [type(law).__name__ for _, laws in compilation.ordered for law in laws].count('AlgebraicLaw')
        7
    """

    guesses_: dict[Expression, float]
    numeric_: list[Expression]
    algebraic_: dict[Expression, Expression]
    arguments_: list[Expression]
    solve_: Callable[..., list[float]]

    def __init__(self, relations: Sequence[_Relation], unknowns: Mapping[Expression, Expression]) -> None:
        self.relations = relations
        self.unknowns = unknowns

    @staticmethod
    def _get_held(relation: _Relation, unknowns: Mapping[Expression, Expression]) -> list[Expression]:
        """Return the unknowns a relation holds, in the order of its variables."""
        return [item for item in relation.expression.variables() if item in unknowns]

    @staticmethod
    def _get_matching(relations: Sequence[_Relation], unknowns: Mapping[Expression, Expression]) -> dict[int, int]:
        """Return a maximum matching of relations to unknowns.

        It gives the position of the unknown of every relation.
        """
        positions = {item: position for position, item in enumerate(unknowns)}
        graph = nx.Graph()
        graph.add_nodes_from(('relation', position) for position in range(len(relations)))
        graph.add_nodes_from(('unknown', position) for position in positions.values())
        graph.add_edges_from(
            (('relation', position), ('unknown', positions[item]))
            for position, relation in enumerate(relations)
            for item in EquationBlocksLowering._get_held(relation, unknowns)
        )
        top = [('relation', position) for position in range(len(relations))]
        matching = bipartite.hopcroft_karp_matching(graph, top_nodes=top)
        return {node[1]: matching[node][1] for node in top if node in matching}

    @staticmethod
    def _get_overdetermined(
        relations: Sequence[_Relation],
        unknowns: Mapping[Expression, Expression],
        matching: Mapping[int, int],
    ) -> list[int]:
        """Return the unmatched relations and those their unknowns are matched to.

        The matching is followed through.
        """
        names = list(unknowns)
        owners = {unknown: position for position, unknown in matching.items()}
        found = [position for position in range(len(relations)) if position not in matching]
        stack = list(found)
        while stack:
            for item in EquationBlocksLowering._get_held(relations[stack.pop()], unknowns):
                owner = owners.get(names.index(item))
                if owner is not None and owner not in found:
                    found.append(owner)
                    stack.append(owner)
        return sorted(found)

    @staticmethod
    def _get_state_names(relation: _Relation, items: Sequence[Expression]) -> list[str]:
        """Return the dotted names of the states that symbols of a relation stand for."""
        references = relation.spec.references
        return [references[item].entry.name for item in items if item in references and references[item].is_state]

    @staticmethod
    def _get_structure_issues(
        relations: Sequence[_Relation],
        unknowns: Mapping[Expression, Expression],
        references: Mapping[Expression, ReferenceSpec],
        label: str,
    ) -> list[IssueSpec]:
        """Return the issue of relations and unknowns that no matching pairs.

        The issue names the states they constrain.

        Args:
            relations:
                The relations of the system, with those of one target added into one.

            unknowns:
                The state each unknown stands for.

            references:
                The references of the symbols of the system.

            label:
                The label of the system, the path of an issue without an equation to name.

        Returns:
            issues:
                The issue, or none for a system whose relations and unknowns all match.
        """
        matching = EquationBlocksLowering._get_matching(relations, unknowns)
        names = list(unknowns)
        free = [unknowns[names[position]] for position in range(len(names)) if position not in matching.values()]
        if len(matching) == len(relations) and not free:
            return []
        constraining = [
            relations[item] for item in EquationBlocksLowering._get_overdetermined(relations, unknowns, matching)
        ]
        constrained = {
            name
            for relation in constraining
            for name in EquationBlocksLowering._get_state_names(
                relation,
                [item for item in relation.expression.variables() if item not in unknowns],
            )
        }
        undetermined = sorted({references[item].entry.name for item in free})
        message = (
            f'{len(relations)} equations cannot be matched to {len(unknowns)} unknowns. The equations constrain the '
            f'states {", ".join(sorted(constrained)) or "none"}, and no equation determines '
            f'{", ".join(undetermined) or "any further unknown"}.'
        )
        fix = (
            'Add a compliant element, such as a resistance, between the constrained states, and give every unknown one '
            'equation.'
        )
        path = constraining[0].spec.owner if constraining else label
        return [IssueSpec('structural-singularity', path, message, fix)]

    @staticmethod
    def _get_blocks(
        relations: Sequence[_Relation],
        unknowns: Mapping[Expression, Expression],
        matching: Mapping[int, int],
    ) -> list[list[int]]:
        """Return the positions of the relations of every block.

        Each block comes after the blocks it reads.
        """
        positions = {item: position for position, item in enumerate(unknowns)}
        owners = {unknown: position for position, unknown in matching.items()}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(relations)))
        graph.add_edges_from(
            (owners[positions[item]], position)
            for position, relation in enumerate(relations)
            for item in EquationBlocksLowering._get_held(relation, unknowns)
            if owners[positions[item]] != position
        )
        condensed = nx.condensation(graph)
        return [sorted(condensed.nodes[node]['members']) for node in nx.topological_sort(condensed)]

    @staticmethod
    def _get_printed(expressions: Sequence[Expression], arguments: Sequence[Expression], time: Expression) -> _Printed:
        """Return the function of time and arguments that evaluates expressions."""
        function = ExpressionPrinter(time=time, compiled=True).prepare(list(arguments)).run(expressions).code_
        if not callable(function):
            error_msg = 'The printer returns no function of the equations of a block.'
            raise TypeError(error_msg)
        return function

    def _get_guesses_of(
        self,
        specs: Sequence[tuple[KernelSpec, list[BaseLaw | _Relation]]],
    ) -> dict[Expression, float]:
        """Return the initial value of every algebraic unknown.

        It is the starting guess of a nonlinear solve.
        """
        layout = Compilation._get_required(self.compilation_.layout, 'layout')
        initial = np.concatenate([np.zeros(0), *layout.initial_states])
        references = {key: value for spec, _ in specs for key, value in spec.references.items()}
        return {
            item: float(initial[references[item].entry.start + references[item].column])
            for item in self.unknowns
            if item in references
        }

    def _get_solved_linear(
        self,
        block: list[Expression],
        residuals: list[Expression],
    ) -> dict[Expression, Expression] | None:
        """Return the symbolic solution of a block linear in up to six unknowns.

        It is `None` for any other block.
        """
        coefficients = [[residual.diff(item) for item in block] for residual in residuals]
        names = set(block)
        if len(block) > SYMBOLIC_UNKNOWNS or any(
            names & set(entry.variables()) for row in coefficients for entry in row
        ):
            return None
        system = matrix(SR, coefficients)
        determinant = SR(system.det())
        if (determinant if determinant.is_numeric() else determinant.simplify_full()).is_trivial_zero():
            states = ', '.join(str(self.unknowns[item]) for item in block)
            error_msg = f'The equations of {states} cannot be solved for them: the system is singular.'
            raise ValidationError(error_msg)
        zero = dict.fromkeys(block, 0)
        right = vector(-residual.subs(zero) for residual in residuals)
        return dict(zip(block, system.solve_right(right, check=False), strict=True))

    def _get_numeric_block(self, block: list[Expression], residuals: list[Expression]) -> _NumericBlock:
        """Return the numeric solve of a block.

        It is linear when its coefficients hold none of its unknowns.
        """
        names = set(block)
        linear = not any(names & set(residual.diff(item).variables()) for residual in residuals for item in block)
        return _NumericBlock(tuple(block), tuple(residuals), linear)

    def _get_linear_solution(
        self,
        block: _NumericBlock,
        arguments: Sequence[Expression],
        time: Expression,
    ) -> _BlockSolution:
        """Return the function that solves a linear block with a numeric linear solve."""
        size = len(block.unknowns)
        zero = dict.fromkeys(block.unknowns, 0)
        entries = [residual.diff(item) for residual in block.residuals for item in block.unknowns]
        offsets = [residual.subs(zero) for residual in block.residuals]
        function = self._get_printed([*entries, *offsets], arguments, time)

        def solve(t: float, known: list[float]) -> list[float]:
            values = np.asarray(function(t, *known), dtype=float)
            try:
                found = np.linalg.solve(values[: size * size].reshape(size, size), -values[size * size :])
            except np.linalg.LinAlgError as error:
                names = ', '.join(map(str, block.unknowns))
                error_msg = f'The linear equations of {names} are singular at time {t}. Check the values of the system.'
                raise SolverError(error_msg) from error
            return [float(item) for item in found]

        return solve

    def _get_root_solution(
        self,
        block: _NumericBlock,
        arguments: Sequence[Expression],
        time: Expression,
    ) -> _BlockSolution:
        """Return the function that solves a nonlinear block.

        It starts from the values of its last evaluation, or one away.
        """
        size = len(block.unknowns)
        names = [*arguments, *block.unknowns]
        residual = self._get_printed(block.residuals, names, time)
        derivatives = [equation.diff(item) for equation in block.residuals for item in block.unknowns]
        jacobian = self._get_printed(derivatives, names, time)
        last = [np.asarray([self.guesses_.get(item, 0.0) for item in block.unknowns], dtype=float)]

        def solve(t: float, known: list[float]) -> list[float]:
            def residuals(values: Array) -> Array:
                return np.asarray(residual(t, *known, *values), dtype=float)

            def slopes(values: Array) -> Array:
                return np.asarray(jacobian(t, *known, *values), dtype=float).reshape(size, size)

            with np.errstate(all='ignore'):
                result = root(residuals, last[0], jac=slopes, tol=ROOT_TOLERANCE)
                if not np.all(np.abs(result.fun) <= RESIDUAL_TOLERANCE):
                    result = root(residuals, last[0] + 1.0, jac=slopes, tol=ROOT_TOLERANCE)
            if not np.all(np.abs(result.fun) <= RESIDUAL_TOLERANCE):
                states = ', '.join(map(str, block.unknowns))
                error_msg = (
                    f'The nonlinear equations of {states} have no solution near the last values at time {t}: '
                    f'{result.message} Give the algebraic states initial values closer to a solution.'
                )
                raise SolverError(error_msg)
            last[0] = result.x
            return [float(item) for item in result.x]

        return solve

    def _get_chained_solution(self, blocks: Sequence[_NumericBlock], time: Expression) -> None:
        """Set `arguments_` and `solve_`.

        `solve_` is the function of the arguments that returns the numeric unknowns.
        """
        solved = {item for block in blocks for item in block.unknowns}
        self.arguments_ = sorted(
            symbols_of([item for block in blocks for item in block.residuals]) - solved - {time},
            key=str,
        )
        solutions = []
        before = list(self.arguments_)
        for block in blocks:
            make = self._get_linear_solution if block.linear else self._get_root_solution
            solutions.append(make(block, before, time))
            before += block.unknowns
        memory: list[tuple[tuple[float, ...], list[float]]] = []

        def solve(t: float, *values: float) -> list[float]:
            key = (t, *values)
            if memory and memory[0][0] == key:
                return memory[0][1]
            known, found = list(values), []
            for solution in solutions:
                part = solution(t, known)
                known += part
                found += part
            memory[:] = [(key, found)]
            return found

        self.solve_ = solve

    def _get_numeric_value(self, expressions: Sequence[Expression], time: Expression) -> CallSpec:
        """Return the call that evaluates expressions at the numeric block unknowns."""
        chained = list(self.arguments_)
        own = sorted(symbols_of(expressions) - set(self.numeric_) - set(chained) - {time}, key=str)
        function = self._get_printed(expressions, [*chained, *own, *self.numeric_], time)
        count, solve = len(chained), self.solve_

        def evaluate(t: float, *values: float) -> list[npt.ArrayLike]:
            return function(t, *values, *solve(t, *values[:count]))

        return CallSpec(evaluate, (*chained, *own))

    def _get_value_of(self, value: Expression, time: Expression) -> Expression | CallSpec:
        """Return a value that reads the numeric unknowns as a call.

        Any other value is returned as it is.
        """
        return self._get_numeric_value([value], time) if set(self.numeric_) & set(value.variables()) else value

    def _get_substituted_vector(self, value: FreeModuleElement, time: Expression) -> object:
        """Return a vector value reading the solutions.

        It is a call when it reads the numeric unknowns.
        """
        items = [SR(item) for item in value]
        if not any(set(self.algebraic_) & set(item.variables()) for item in items):
            return value
        replaced = [item.subs(self.algebraic_) for item in items]
        if any(set(self.numeric_) & set(item.variables()) for item in replaced):
            return self._get_numeric_value(replaced, time)
        return vector(replaced)

    def _get_substituted(self, value: object, time: Expression, owner: str) -> object:
        """Return the value of a law or a jump, reading the solutions.

        The law is a rate, noise, update, or boundary law.
        """
        if isinstance(value, FreeModuleElement):
            return self._get_substituted_vector(value, time)
        if isinstance(value, CallSpec) and value.arguments:
            return replace(value, arguments=tuple(self._get_symbolic_only(SR(item), owner) for item in value.arguments))
        if isinstance(value, Expression) and set(self.algebraic_) & set(value.variables()):
            return self._get_value_of(value.subs(self.algebraic_), time)
        return value

    def _get_symbolic_only(self, value: Expression, owner: str) -> Expression:
        """Return a value with the symbolic solutions.

        A value that a numeric solve determines raises an error.
        """
        replaced = value.subs(self.algebraic_) if set(self.algebraic_) & set(value.variables()) else value
        if set(self.numeric_) & set(replaced.variables()):
            names = ', '.join(sorted(str(item) for item in set(self.numeric_) & set(replaced.variables())))
            error_msg = (
                f'A law of {owner} reads {names} in a function argument, an event guard, or a reset, but a nonlinear '
                'solve determines these algebraic states at every evaluation. Read states of an element instead.'
            )
            raise ValidationError(error_msg)
        return replaced

    def _copy_with_solutions(self, law: BaseLaw, owner: str, time: Expression) -> BaseLaw:
        """Return a law whose values read the solutions of the algebraic states.

        It is the law itself when it reads none.
        """
        if isinstance(law, KERNEL_FREE_LAWS):
            return law
        if isinstance(law, JumpLaw):
            rate = self._get_substituted(law.rate, time, owner)
            return law if rate is law.rate else BasePass._get_replaced(law, rate=rate)
        if isinstance(law, EventLaw | RenewalLaw):
            return self._copy_with_event_solutions(law, owner)
        if isinstance(law, RateLaw | NoiseLaw | UpdateLaw | _BoundaryLaw):
            value = self._get_substituted(law.value, time, owner)
            return law if value is law.value else BasePass._get_replaced(law, value=value)
        return law

    def _copy_with_event_solutions(self, law: EventLaw | RenewalLaw, owner: str) -> BaseLaw:
        """Return an event or renewal that reads the symbolic solutions.

        Its guard, waiting time, and reset read them.
        """
        name = 'waiting' if isinstance(law, RenewalLaw) else 'guard'
        value = getattr(law, name)
        changes: dict[str, object] = {}
        if isinstance(value, Expression):
            changes[name] = self._get_symbolic_only(value, owner)
        if law.reset is not None:
            changes['reset'] = {
                target: self._get_symbolic_only(item, owner) if isinstance(item, Expression) else item
                for target, item in law.reset.items()
            }
        return BasePass._get_replaced(law, **changes) if changes else law

    def _get_kept_spec(
        self,
        spec: KernelSpec,
        laws: Sequence[BaseLaw],
        references: Mapping[Expression, ReferenceSpec],
        time: Expression,
    ) -> tuple[KernelSpec, list[BaseLaw]]:
        """Return a spec and its laws read through the solutions.

        The spec has the references of the symbols they now read.
        """
        kept = [self._copy_with_solutions(law, spec.owner, time) for law in laws]
        if all(new is old for new, old in zip(kept, laws, strict=True)):
            return spec, kept
        return replace(spec, references={**references, **spec.references}), kept

    def _solve_blocks(
        self,
        relations: Sequence[_Relation],
        blocks: Sequence[Sequence[int]],
        matching: Mapping[int, int],
    ) -> tuple[dict[Expression, Expression], list[_NumericBlock]]:
        """Return the solution of every unknown and the numeric blocks.

        The solutions are in known symbols and numeric unknowns.
        """
        names = list(self.unknowns)
        solutions: dict[Expression, Expression] = {}
        numeric: list[_NumericBlock] = []
        for block in blocks:
            unknowns = [names[matching[position]] for position in block]
            residuals = [relations[position].expression for position in block]
            residuals = [item.subs(solutions) if solutions else item for item in residuals]
            solved = self._get_solved_linear(unknowns, residuals)
            if solved is None:
                numeric.append(self._get_numeric_block(unknowns, residuals))
            else:
                solutions.update(solved)
        return solutions, numeric

    @staticmethod
    def _get_member_groups(relations: Sequence[_Relation]) -> tuple[str | None, tuple[str, ...]]:
        """Return the group and the aligned groups that the solved laws run over.

        A constraint of an interaction on a group, such as `y == a x^2` for every point of a group, holds for every
        member, so its solution reads the values of each member in its own row.
        """
        groups = list(dict.fromkeys(relation.spec.group for relation in relations if relation.spec.group is not None))
        aligned = dict.fromkeys(item for relation in relations for item in relation.spec.aligned)
        if not groups:
            return None, ()
        return groups[0], tuple(dict.fromkeys([*groups[1:], *aligned]))

    def _run(self, inputs: Sequence[tuple[KernelSpec, list[BaseLaw | _Relation]]] | None) -> None:
        """Set `lowered_`, the specs without their relations and a spec of unknowns.

        The last spec holds the laws of the unknowns.

        Args:
            inputs:
                The laws and relations of every spec.
        """
        specs = list(inputs or [])
        time = Compilation._get_required(self.compilation_.layout, 'layout').time
        relations = self.relations
        self.guesses_ = self._get_guesses_of(specs)
        matching = self._get_matching(relations, self.unknowns)
        solutions, numeric = self._solve_blocks(
            relations,
            self._get_blocks(relations, self.unknowns, matching),
            matching,
        )
        self.numeric_ = [item for block in numeric for item in block.unknowns]
        self._get_chained_solution(numeric, time)
        references = {key: value for relation in relations for key, value in relation.spec.references.items()}
        self.algebraic_ = {
            item: solutions.get(item, item) for item, state in self.unknowns.items() if item.is_trivially_equal(state)
        }
        owner = ', '.join(dict.fromkeys(relation.spec.owner for relation in relations))
        rows = max((relation.spec.rows for relation in relations), default=1)
        group, aligned = self._get_member_groups(relations)
        merged = KernelSpec(owner, SimpleNamespace(), references, rows, group=group, aligned=aligned)
        laws: list[BaseLaw] = []
        for item, state in self.unknowns.items():
            value = self._get_value_of(SR(solutions.get(item, item)), time)
            laws.append(AlgebraicLaw(state, value) if item.is_trivially_equal(state) else RateLaw(state, value))
        kept = [
            self._get_kept_spec(spec, [law for law in items if not isinstance(law, _Relation)], references, time)
            for spec, items in specs
        ]
        self.lowered_ = [*kept, (merged, laws)]
