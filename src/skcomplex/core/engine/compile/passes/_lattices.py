"""Compile the laws that update the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Sequence
from dataclasses import replace

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression
from sage.ext.fast_callable import fast_callable
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import (
    BaseLaw,
    LinearRuleSpec,
    MoveSpec,
    ObjectiveSpec,
    OuterTotalisticSpec,
    SamplerSpec,
    TableSpec,
    TotalisticSpec,
    UpdateLaw,
)
from ....components.laws._local_rule import _LocalRuleLaw
from ....components.laws._sample import _SampleLaw
from ....components.laws.utils._reductions import neighborhoods_in
from ....components.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CellComplexSpace,
    CofaceNeighborhood,
    LatticeSpace,
    NetworkSpace,
)
from ....errors import SpaceError, ValidationError
from ....types import Array, Indices
from ..._utils import symbol
from ...enums import ReadKind
from ...problems._arguments_spec import _ArgumentsSpec
from ...problems._cascade_kernel import CascadeKernel
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._drive_kernel import DriveKernel
from ...problems._expression_printer import ExpressionPrinter
from ...problems._kernel_spec import KernelSpec
from ...problems._lattice_state_spec import _LatticeStateSpec
from ...problems._lookup_spec import _LookupSpec
from ...problems._monte_carlo_kernel import MonteCarloKernel
from ...problems._rule_kernel import RuleKernel
from .._array_update_law import ArrayUpdateLaw
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation
from .._issue_spec import IssueSpec
from .._lattice_context_spec import _LatticeContextSpec
from .._laws import LATTICE_LAWS
from .._relaxation_law import RelaxationLaw
from ._cells import DEGREES

COLORING_STRATEGY = 'largest_first'


type LatticeStep = RuleKernel | DriveKernel | CascadeKernel | MonteCarloKernel
type LoweredObjective = tuple[
    Indices,
    tuple[Indices, ...],
    list[Expression],
    list[Expression],
    Expression,
    tuple[scipy.sparse.csr_array, Array] | None,
]


class LatticeProblemPass(BasePass):
    """A pass that adds the steps of the laws of a lattice to the problem.

    It compiles the local rules of a lattice, such as the rule of the Game of Life, the drives and cascades of a
    sandpile, and the Monte Carlo moves of a lattice of spins, into the steps that the lattice solver runs.

    Read more in the user guide, under Extend the library.

    See Also:
        RelaxationLaw: Relaxes a lattice until it is stable.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> compilation = PassCompiler().prepare(IsingSystem()).run().compilation_
        >>> [type(step).__name__ for step in compilation.problem.lattice]
        ['MonteCarloKernel']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the lattice steps to the problem."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        members = declarations.members
        fused = BaseCompiler._resolve_backend(compilation.backend, members)._get_name() == 'numba'
        compilation.problem = replace(
            Compilation._get_required(compilation.problem, 'problem'),
            lattice=self._compile_lattice(
                Compilation._get_required(compilation.reduced, 'reduced'),
                declarations,
                fused,
            ),
        )

    @staticmethod
    def _get_lattice_state(
        context: _LatticeContextSpec,
        item: Expression,
        law: str,
        *,
        components: bool = False,
    ) -> _LatticeStateSpec:
        """Return the lattice state of a target symbol, or raise an error naming the law.

        Only a law that changes every component of a member, a drive or a local rule, may target a state of several
        components.
        """
        if item not in context.states:
            error_msg = f'{law} of {context.owner} targets {item}, which is not a state of a lattice group.'
            raise ValidationError(error_msg)
        state = context.states[item]
        if state.components > 1 and not components:
            error_msg = (
                f'{law} of {context.owner} targets {state.name}, which holds {state.components} components per member. '
                'Change it with a drive or a function local rule, which receive the components of every member on a '
                'last axis.'
            )
            raise ValidationError(error_msg)
        return state

    @staticmethod
    def _get_keywords(context: _LatticeContextSpec) -> _ArgumentsSpec:
        """Return every parameter its component reads, which a rule receives by keyword."""
        names, starts, sizes = [], [], []
        for name, is_state, start, size in context.arguments.values():
            if not is_state:
                names.append(name)
                starts.append(start)
                sizes.append(size)
        return _ArgumentsSpec(tuple(names), (False,) * len(names), tuple(starts), tuple(sizes))

    @staticmethod
    def _get_lattice_context(spec: KernelSpec, declarations: DeclarationsSpec, fused: bool) -> _LatticeContextSpec:
        """Return the lattice states, arguments, and neighbour symbols of a component.

        They are those of the symbols the component reads.
        """
        states, arguments, others = {}, {}, {}
        for item, reference in spec.references.items():
            entry = reference.entry
            owner, _, variable = entry.name.rpartition('.')
            block = declarations.blocks.get(owner)
            space = block.node.space if block is not None else None
            is_lattice = block is not None and block.node.members is not None and not block.node.is_agent
            if reference.is_state and is_lattice and isinstance(space, BaseSpace) and space._has_sites():
                part = (getattr(block.node.component, 'on', None) if block is not None else None) or 'sites'
                rows = None if part == 'sites' else entry.rows
                states[item] = _LatticeStateSpec(entry.name, entry.start, space, part, rows, entry.columns)
            if not entry.member_shape:
                arguments[item] = (variable, reference.is_state, entry.start, entry.size)
            if reference.source is ReadKind.OTHER:
                others[item] = entry.name
        return _LatticeContextSpec(spec.owner, states, arguments, fused, others)

    @staticmethod
    def _get_monte_carlo_step(spec: KernelSpec, law: _SampleLaw, context: _LatticeContextSpec) -> LatticeStep:
        """Return the Monte Carlo sampling of a sample.

        It uses Metropolis acceptance at `intensity` by default.
        """
        sampler = law.sampler
        if sampler is None:
            if not hasattr(spec.namespace, 'intensity'):
                error_msg = f'{spec.owner} declares a sample without a sampler and no parameter `intensity`.'
                raise ValidationError(error_msg)
            sampler = SamplerSpec(intensity=spec.namespace.intensity)
        return LatticeProblemPass._compile_monte_carlo(law.objective, law.move, sampler, context)

    @staticmethod
    def _get_lattice_step(spec: KernelSpec, law: BaseLaw, context: _LatticeContextSpec) -> LatticeStep | None:
        """Return the step of a lattice law, or `None` for a law of another kind."""
        if isinstance(law, _LocalRuleLaw):
            return LatticeProblemPass._compile_rule(law, context)
        if isinstance(law, ArrayUpdateLaw):
            return LatticeProblemPass._compile_drive(law, context)
        if isinstance(law, RelaxationLaw):
            return LatticeProblemPass._compile_cascade(law, context)
        if isinstance(law, _SampleLaw):
            return LatticeProblemPass._get_monte_carlo_step(spec, law, context)
        return None

    @staticmethod
    def _compile_lattice(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: DeclarationsSpec,
        fused: bool,
    ) -> tuple[LatticeStep, ...]:
        """Return the steps of lattice laws in declaration order.

        They are local rules, drives, cascades, and samples.
        """
        steps: list[LatticeStep] = []
        for spec, laws in specs:
            if not any(isinstance(law, LATTICE_LAWS) for law in laws):
                continue
            context = LatticeProblemPass._get_lattice_context(spec, declarations, fused)
            found = [LatticeProblemPass._get_lattice_step(spec, law, context) for law in laws]
            steps += [step for step in found if step is not None]
        return tuple(steps)

    @staticmethod
    def _convert_to_table(rule: object) -> object:
        """Return a rule code as its table.

        The table comes from its `get_table()` method, and any other rule is unchanged.
        """
        convert = getattr(rule, 'get_table', None)
        return convert() if not callable(rule) and callable(convert) else rule

    @staticmethod
    def _get_calls(expression: Expression, name: str) -> list[Expression]:
        """Return the calls of a symbolic function by name inside an expression.

        The outermost calls come first.
        """
        found, stack = [], [SR(expression)]
        while stack:
            item = stack.pop()
            if item.operator() is None:
                continue
            if getattr(item.operator(), 'name', lambda: '')() == name:
                found.append(item)
                continue
            stack.extend(item.operands())
        return found

    @staticmethod
    def _get_center_columns(neighborhood: BaseNeighborhood, space: BaseSpace) -> int:
        """Return the number of columns of a neighbourhood that read the site itself."""
        return neighborhood._count_centers(space.n_dim_)

    @staticmethod
    def _check_checkerboard(
        space: BaseSpace,
        neighborhood: BaseNeighborhood,
        owner: str,
    ) -> LatticeSpace:
        """Return the square lattice of a checkerboard update.

        An error is raised when neighbours change together.
        """
        misfit = LatticeProblemPass._get_checkerboard_misfit(space, neighborhood)
        if not isinstance(space, LatticeSpace) or misfit == 'square':
            error_msg = (
                f"{owner} uses a checkerboard schedule, which needs a square lattice. Use the 'colored' schedule, "
                'which updates the classes of a proper coloring of the neighbour graph in turn.'
            )
            raise ValidationError(error_msg)
        if misfit == 'parity':
            error_msg = (
                f'{owner} uses a checkerboard schedule, which needs neighbours on the other sublattice and even sizes '
                "on periodic lattices. Choose another schedule in the solver, such as "
                "solver=MetropolisSolver(moves=..., schedule='random-sequential'), or leave the schedule 'auto'."
            )
            raise ValidationError(error_msg)
        return space

    @staticmethod
    def _get_checkerboard_misfit(space: BaseSpace, neighborhood: BaseNeighborhood) -> str | None:
        """Return why a checkerboard update does not fit a lattice, or `None` when it fits.

        It is `'square'` for a lattice that is not square and `'parity'` for neighbours or sizes of the wrong parity.
        """
        offsets = neighborhood._get_offsets(space.n_dim_) if isinstance(space, LatticeSpace) else None
        if not isinstance(space, LatticeSpace) or space.geometry != 'square' or offsets is None:
            return 'square'
        odd = all(sum(offset) % 2 for offset in offsets)
        even = space._get_edges() != 'periodic' or all(size % 2 == 0 for size in space.shape_)
        return None if odd and even else 'parity'

    @staticmethod
    def _resolve_schedule(sampler: SamplerSpec, space: BaseSpace, over: object, cofaces: bool) -> SamplerSpec:
        """Return the sampler with an `'auto'` schedule fitted to the lattice.

        Sums over cofaces take the colored schedule, and other sums take the checkerboard when it fits and the random
        sequential order otherwise.
        """
        if sampler.schedule != 'auto':
            return sampler
        if cofaces:
            return replace(sampler, schedule='colored')
        fits = isinstance(over, BaseNeighborhood) and LatticeProblemPass._get_checkerboard_misfit(space, over) is None
        return replace(sampler, schedule='checkerboard' if fits else 'random-sequential')

    @staticmethod
    def _get_coloring(size: int, rows: Indices, columns: Indices) -> tuple[Indices, ...]:
        """Return the classes of a proper coloring of the graph of some members.

        The graph joins the given pairs.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(zip(rows.tolist(), columns.tolist(), strict=True))
        colors = nx.greedy_color(graph, strategy=COLORING_STRATEGY)
        found = np.asarray([colors[site] for site in range(size)], dtype=np.intp)
        return tuple(np.flatnonzero(found == color) for color in range(int(found.max(initial=-1)) + 1))

    @staticmethod
    def _get_color_classes(table: Indices) -> tuple[Indices, ...]:
        """Return the classes of a proper coloring of the graph of the table rows.

        The graph joins every site to the sites of its table row.
        """
        size = table.shape[0]
        rows = np.repeat(np.arange(size), table.shape[1])
        columns = table.ravel()
        real = (columns != size) & (columns != rows)
        return LatticeProblemPass._get_coloring(size, rows[real], columns[real])

    @staticmethod
    def _get_coface_classes(boundary: scipy.sparse.csr_array) -> tuple[Indices, ...]:
        """Return the classes of a proper coloring of the graph of shared cofaces.

        The graph joins the cells that share a coface.
        """
        shared = (abs(boundary).T @ abs(boundary)).tocoo()
        real = shared.row != shared.col
        return LatticeProblemPass._get_coloring(boundary.shape[1], shared.row[real], shared.col[real])

    @staticmethod
    @functools.cache
    def _get_sublattices(space: LatticeSpace) -> tuple[Indices, Indices]:
        """Return the sites of even and of odd coordinate sum."""
        colors = np.indices(space.shape_).reshape(space.n_dim_, -1).sum(axis=0) % 2
        return np.flatnonzero(colors == 0), np.flatnonzero(colors == 1)

    @staticmethod
    def _schedule_classes(
        schedule: str,
        space: BaseSpace,
        neighborhood: BaseNeighborhood,
        table: Indices,
        owner: str,
    ) -> tuple[Indices, ...]:
        """Return the classes of sites a checkerboard or colored schedule updates in turn.

        It is none for other schedules.
        """
        if schedule == 'checkerboard':
            return LatticeProblemPass._get_sublattices(
                LatticeProblemPass._check_checkerboard(space, neighborhood, owner),
            )
        if schedule == 'colored':
            return LatticeProblemPass._get_color_classes(table)
        return ()

    @staticmethod
    def _get_degrees_of(table: Indices, neighborhood: BaseNeighborhood, space: BaseSpace) -> Array:
        """Return the number of real neighbours of every site, besides the site itself."""
        centers = LatticeProblemPass._get_center_columns(neighborhood, space)
        return ((table != table.shape[0]).sum(axis=1) - centers).astype(float)

    @staticmethod
    def _get_index_table(
        space: BaseSpace,
        neighborhood: BaseNeighborhood,
        owner: str,
    ) -> Indices:
        """Return the neighbour table of a lattice or a network.

        An error naming the law that reads it is raised without one.
        """
        try:
            return space._get_index_table(neighborhood)
        except SpaceError as error:
            error_msg = f'{owner}: {error}'
            raise SpaceError(error_msg) from error
        except (ValueError, ValidationError) as error:
            error_msg = f'{owner}: {error}'
            raise ValidationError(error_msg) from error

    @staticmethod
    def _get_lookup_of(rule: object, n_offsets: int, owner: str) -> _LookupSpec | None:
        """Return the lookup array of a table or totalistic rule.

        It is `None` for a function rule.
        """
        if isinstance(rule, TableSpec):
            base = rule.n_states
            values = np.full(base**n_offsets, float(rule.default))
            for pattern, value in rule.mapping.items():
                if len(pattern) != n_offsets:
                    error_msg = f'{owner} has a table pattern {pattern} for a neighbourhood of {n_offsets} cells.'
                    raise ValidationError(error_msg)
                values[int(np.ravel_multi_index(tuple(int(item) for item in pattern), (base,) * n_offsets))] = value
            return _LookupSpec('table', base, values)
        if isinstance(rule, TotalisticSpec | OuterTotalisticSpec):
            sums = np.arange(n_offsets * (rule.states - 1) + 1)
            if isinstance(rule, TotalisticSpec):
                if callable(rule.rule):
                    values = np.broadcast_to(np.asarray(rule.rule(sums), dtype=float), sums.shape).copy()
                else:
                    values = np.asarray([float(rule.rule.get(int(total), rule.default)) for total in sums])
                return _LookupSpec('totalistic', rule.states, values)
            centers = np.arange(rule.states)[:, None]
            if callable(rule.rule):
                grid = np.broadcast_to(
                    np.asarray(rule.rule(centers, sums[None, :]), dtype=float),
                    (rule.states, sums.size),
                )
            else:
                grid = np.asarray(
                    [
                        [float(rule.rule.get((int(center), int(total)), rule.default)) for total in sums]
                        for center in range(rule.states)
                    ],
                )
            return _LookupSpec('outer', rule.states, grid.ravel().copy(), sums.size)
        return None

    @staticmethod
    def _lower_neighbor_sums(
        expression: Expression,
        target: Expression,
        *,
        objective: ObjectiveSpec,
        context: _LatticeContextSpec,
        neighbors: Sequence[Expression],
        masks: Sequence[Expression],
    ) -> Expression:
        """Return an expression with every neighbour sum written as a masked sum.

        The sum runs over the columns of the table.
        """
        owner = f'The Monte Carlo laws of {context.owner}'
        moved = context.states[target].name
        replacements = {}
        for reduction in neighborhoods_in(expression):
            if reduction.kind != 'sum' or reduction.over not in (None, objective.over):
                error_msg = (
                    f'{owner} reduce {reduction.call}, but a local objective only sums over the neighbourhood of the '
                    f'objective, {objective.over!r}.'
                )
                raise ValidationError(error_msg)
            inner = SR(reduction.argument) if reduction.weight is None else SR(reduction.weight) * reduction.argument
            reads = [item for item in inner.variables() if item in context.others]
            for item in reads:
                if context.others[item] != moved:
                    error_msg = f'{owner} read the neighbours of {item}, but only the moved state {target} has them.'
                    raise ValidationError(error_msg)
            columns = [inner.subs(dict.fromkeys(reads, neighbor)) for neighbor in neighbors]
            replacements[reduction.call] = sum(
                (mask * column for mask, column in zip(masks, columns, strict=True)),
                SR(0),
            )
        return SR(expression).subs(replacements) if replacements else SR(expression)

    @staticmethod
    def _lower_coface_sums(
        expression: Expression,
        target: Expression,
        columns: tuple[Sequence[Expression], Sequence[Expression], Sequence[Expression]],
        owner: str,
    ) -> Expression:
        """Return an expression with every coface sum written over the cofaces.

        The sum is written over the rests, the signs, and the masks of the cofaces. Inside a coface, the exterior
        derivative of the moved state is its sign times the state plus the rest of the oriented sum around the coface.

        Args:
            expression:
                The local objective.

            target:
                The moved state.

            columns:
                The symbols of the rest, the sign, and the mask of every coface.

            owner:
                The name of the laws for an error.

        Returns:
            lowered:
                The local objective without reductions.

        Raises:
            ValidationError:
                A reduction is not a sum over the cofaces, or the exterior derivative reads another expression.
        """
        replacements = {}
        for reduction in neighborhoods_in(expression):
            if reduction.kind != 'sum' or not isinstance(reduction.over, CofaceNeighborhood):
                error_msg = f'{owner} reduce {reduction.call}, but a local objective on cofaces only sums over them.'
                raise ValidationError(error_msg)
            inner = SR(reduction.argument) if reduction.weight is None else SR(reduction.weight) * reduction.argument
            calls = LatticeProblemPass._get_calls(inner, 'exterior_derivative')
            for call in calls:
                if call.operands()[0] != target:
                    error_msg = f'{owner} read {call}, but inside a coface only the moved state {target} has a sum.'
                    raise ValidationError(error_msg)
            replacements[reduction.call] = sum(
                (
                    mask * inner.subs(dict.fromkeys(calls, sign * target + rest))
                    for rest, sign, mask in zip(*columns, strict=True)
                ),
                SR(0),
            )
        return SR(expression).subs(replacements) if replacements else SR(expression)

    @staticmethod
    def _get_coface_tables(
        target: _LatticeStateSpec,
        owner: str,
    ) -> tuple[Indices, Array, scipy.sparse.csr_array]:
        """Return the cofaces of the moved cells, their signs, and incidences.

        The incidence is that of the cells and the cofaces.
        """
        space = target.space
        degree = DEGREES[target.part]
        if not isinstance(space, CellComplexSpace) or degree >= len(space.incidences_):
            error_msg = f'{owner} sum over cofaces, which needs a state on the sites or edges of a CellComplexSpace.'
            raise ValidationError(error_msg)
        return space.cofaces_[degree], space.orientations_[degree], space.incidences_[degree]

    @staticmethod
    def _compile_drive(law: ArrayUpdateLaw, context: _LatticeContextSpec) -> DriveKernel:
        """Return a drive compiled for its lattice state.

        Args:
            law:
                The drive.

            context:
                The lattice states and symbols of its component.

        Returns:
            drive:
                The compiled drive.
        """
        targets = law._get_targets()
        state = LatticeProblemPass._get_lattice_state(context, targets[0], 'A drive', components=True)
        if len(targets) != state.components:
            error_msg = (
                f'A drive of {context.owner} targets {len(targets)} of the {state.components} components of '
                f'{state.name}. Target the whole state.'
            )
            raise ValidationError(error_msg)
        return DriveKernel(
            context.owner,
            state,
            law.function,
            LatticeProblemPass._get_keywords(context),
            stochastic=law.stochastic,
            score=law.score,
        ).prepare()

    @staticmethod
    def _compile_rule(law: _LocalRuleLaw, context: _LatticeContextSpec) -> RuleKernel:
        """Return a local rule compiled for its lattice.

        Args:
            law:
                The local rule.

            context:
                The lattice states and symbols of its component.

        Returns:
            rule:
                The compiled rule.
        """
        owner = f'The local rule of {context.owner}'
        targets = tuple(
            LatticeProblemPass._get_lattice_state(
                context,
                LatticeProblemPass._get_first_component(item),
                'A local rule',
                components=True,
            )
            for item in law._get_targets()
        )
        reads = tuple(
            LatticeProblemPass._get_lattice_state(
                context,
                LatticeProblemPass._get_first_component(item),
                'A local rule',
                components=True,
            )
            for item in (law.reads or law._get_targets())
        )
        spaces = {id(state.space) for state in (*targets, *reads)}
        if len(spaces) > 1:
            error_msg = f'{owner} reads and updates states of different lattices.'
            raise ValidationError(error_msg)
        space = targets[0].space
        translations = LatticeProblemPass._get_translations_of(law, space, reads, owner)
        table = (
            LatticeProblemPass._get_index_table(space, law.over, owner)
            if translations is None
            else translations.reshape(translations.shape[0], -1)
        )
        n_offsets = table.shape[1] if translations is None else translations.shape[2]
        rule = (
            law._get_linear_values
            if isinstance(law.rule, LinearRuleSpec)
            else LatticeProblemPass._convert_to_table(law.rule)
        )
        if isinstance(space, NetworkSpace) and isinstance(rule, TableSpec):
            error_msg = (
                f'{owner} uses a table of ordered patterns on a network, whose sites have different degrees. Use a '
                'totalistic or an outer totalistic rule, or a function rule.'
            )
            raise ValidationError(error_msg)
        lookup = LatticeProblemPass._get_lookup_of(rule, n_offsets, owner)
        several = len(targets) > 1 or len(reads) > 1
        if lookup is not None and (several or any(state.components > 1 for state in (*targets, *reads))):
            error_msg = f'{owner} uses a lookup rule, which updates and reads a single state of one component.'
            raise ValidationError(error_msg)
        function = None
        if lookup is None:
            if not callable(rule):
                error_msg = (
                    f'{owner} has the rule {law.rule!r}, which is not a function, a Table, a totalistic rule, or a '
                    'rule code with a get_table() method.'
                )
                raise ValidationError(error_msg)
            function = rule
        reads_degree = function is not None and translations is None
        if translations is not None and law.schedule in {'checkerboard', 'colored'}:
            error_msg = (
                f'{owner} reads a memory translated by the values of a state, whose neighbours move with the '
                "configuration, so it has no coloring. Use the 'synchronous', 'random-sequential', or 'random-site' "
                'schedule.'
            )
            raise ValidationError(error_msg)
        return RuleKernel(
            context.owner,
            targets,
            reads,
            table,
            function,
            lookup=lookup,
            schedule=law.schedule,
            stochastic=law.stochastic,
            arguments=LatticeProblemPass._get_keywords(context) if function is not None else _ArgumentsSpec(),
            fused=context.fused and lookup is not None and law.schedule == 'synchronous',
            degrees=LatticeProblemPass._get_degrees_of(table, law.over, space) if reads_degree else None,
            classes=LatticeProblemPass._schedule_classes(law.schedule, space, law.over, table, owner),
            translations=translations,
            name=LatticeProblemPass._get_rule_name(law, context.owner),
        ).prepare()

    @staticmethod
    def _get_rule_name(law: _LocalRuleLaw, owner: str) -> str:
        """Return the name of a local rule that equations show.

        It is the name of a named function, else the label of the law, else the label of the component that owns it.
        """
        name = str(getattr(law.rule, '__name__', '')).lstrip('_')
        if name and name != '<lambda>':
            return name
        return law.label or owner.replace('/', '.').rpartition('.')[2] or type(law.rule).__name__

    @staticmethod
    def _get_first_component(item: object) -> object:
        """Return the symbol of the first component of a state of several components.

        A scalar symbol is returned as it is.
        """
        return item[0] if isinstance(item, FreeModuleElement) else item

    @staticmethod
    def _get_translations_of(
        law: _LocalRuleLaw,
        space: BaseSpace,
        reads: tuple[_LatticeStateSpec, ...],
        owner: str,
    ) -> Indices | None:
        """Return the translation table of a memory translated by the values of a state.

        It is `None` for another memory.

        Args:
            law:
                The local rule.

            space:
                The space of the states of the rule.

            reads:
                The states the rule reads, the first of which holds the group elements.

            owner:
                The name of the rule in error messages.

        Returns:
            translations:
                The site of `g h s` for every site `g`, element `h`, and word `s`, or `None`.

        Raises:
            ValidationError:
                The first read state has several components, or the space translates no memory set.
        """
        if not law.over._is_translated():
            return None
        if reads[0].components > 1:
            error_msg = f'{owner} translates its memory by {reads[0].name}, which holds several components per site.'
            raise ValidationError(error_msg)
        try:
            return space._get_translation_table(law.over)
        except (ValueError, ValidationError, SpaceError) as error:
            error_msg = f'{owner}: {error}'
            raise ValidationError(error_msg) from error

    @staticmethod
    def _compile_part(law: object, context: _LatticeContextSpec) -> RuleKernel | DriveKernel:
        """Return a law of a cascade compiled."""
        if isinstance(law, _LocalRuleLaw):
            return LatticeProblemPass._compile_rule(law, context)
        if isinstance(law, ArrayUpdateLaw):
            return LatticeProblemPass._compile_drive(law, context)
        if isinstance(law, UpdateLaw):
            return LatticeProblemPass._compile_drive(LatticeProblemPass._get_array_update(law, context), context)
        error_msg = f'A cascade of {context.owner} holds {law!r}, which is not a local rule or a drive.'
        raise ValidationError(error_msg)

    @staticmethod
    def _get_array_update(law: UpdateLaw, context: _LatticeContextSpec) -> ArrayUpdateLaw:
        """Return the drive of a cascade, an update of a state by its own value.

        Such a drive, such as `UpdateLaw(h, h + 1)` adding a grain to every site of a sandpile, changes every site by
        the same function of its own value.

        Args:
            law:
                The update law of the drive.

            context:
                The context of the cascade, with the component that owns it.

        Returns:
            update:
                The update of the whole array of the state.

        Raises:
            ValidationError:
                The value reads anything but the state and numbers.
        """
        target, value = law.target, SR(law.value)
        if not isinstance(target, Expression) or not set(value.variables()) <= {target}:
            error_msg = (
                f'A drive of {context.owner} updates {target} by {value}, but an update in a cascade reads its state '
                'alone, such as UpdateLaw(h, h + 1). Give a function of the whole array to read others.'
            )
            raise ValidationError(error_msg)
        step = fast_callable(value, vars=[target], domain=float)
        return ArrayUpdateLaw(target, functools.partial(LatticeProblemPass._apply_step, step), label=law.label)

    @staticmethod
    def _apply_step(step: Callable[[float], float], array: Array, **kwargs: object) -> Array:
        """Return an array with a function of one value applied to every site.

        It ignores the keywords of a rule.
        """
        return np.vectorize(step, otypes=[float])(array)

    @staticmethod
    def _compile_cascade(law: RelaxationLaw, context: _LatticeContextSpec) -> CascadeKernel:
        """Return a cascade compiled for its lattice.

        Args:
            law:
                The cascade.

            context:
                The lattice states and symbols of its component.

        Returns:
            cascade:
                The compiled cascade.
        """
        if not law.relax:
            error_msg = f'A cascade of {context.owner} has no laws to relax with.'
            raise ValidationError(error_msg)
        return CascadeKernel(
            law.label or context.owner,
            tuple(LatticeProblemPass._compile_part(item, context) for item in law.drive),
            tuple(LatticeProblemPass._compile_part(item, context) for item in law.relax),
            tuple(LatticeProblemPass._get_lattice_state(context, item, "A cascade") for item in law._get_targets()),
            law.stable,
            max_iterations=int(law.max_iterations),
            activity=law.activity,
        ).prepare()

    @staticmethod
    def _get_bond_of(
        local: Expression,
        target: Expression,
        neighbours: list[Expression],
        masks: list[Expression],
        owner: str,
    ) -> Expression:
        """Return the half change of the local objective with one neighbour.

        The site flips and then the neighbour. This is `f(s, n) - f(-s, n)` for an objective of pair terms `f` that the
        flip of both values keeps, which gives the Wolff bond probability `1 - exp(min(0, -intensity * (f(-s, n) - f(s,
        n))))`.
        """
        flipped = local.subs({item: -item for item in [target, *neighbours]})
        if not bool((flipped - local).expand().is_trivial_zero()):
            error_msg = (
                f'{owner} make cluster moves on the objective {local}, which changes when every value flips. Cluster '
                'moves take an objective of pair terms that the flip of both values keeps, such as one without an '
                'external field.'
            )
            raise ValidationError(error_msg)
        kept = {**dict.fromkeys(neighbours[1:], 0), **dict.fromkeys(masks, 1)}
        single = local.subs(kept)
        first = neighbours[0]
        mixed = (
            single.subs({target: -target, first: -first})
            - single.subs({target: -target})
            - single.subs({first: -first})
            + single
        )
        return (mixed / 2).expand()

    @staticmethod
    def _lower_neighbors(
        objective: ObjectiveSpec,
        move: MoveSpec,
        sampler: SamplerSpec,
        context: _LatticeContextSpec,
        target: _LatticeStateSpec,
    ) -> LoweredObjective:
        """Return the table, classes, symbols, and expression of a neighbour objective.

        The objective is a local objective over neighbours.
        """
        owner = f'The Monte Carlo laws of {context.owner}'
        over = objective.over
        if not isinstance(over, BaseNeighborhood) or (
            over._get_offsets(target.space.n_dim_) is None
            and over._get_slots() is None
            and not isinstance(over, AdjacencyNeighborhood)
        ):
            error_msg = (
                f'{owner} sum over a {type(over).__name__}. Sum over a lattice, adjacency, or memory neighbourhood.'
            )
            raise ValidationError(error_msg)
        table = LatticeProblemPass._get_index_table(target.space, over, owner)
        classes = LatticeProblemPass._schedule_classes(sampler.schedule, target.space, over, table, owner)
        neighbours = [symbol(f'_neighbor_{index}') for index in range(table.shape[1])]
        masks = [symbol(f'_mask_{index}') for index in range(table.shape[1])]
        expression = LatticeProblemPass._lower_neighbor_sums(
            SR(objective.expression),
            move.target,
            objective=objective,
            context=context,
            neighbors=neighbours,
            masks=masks,
        )
        return table, classes, neighbours, masks, expression, None

    @staticmethod
    def _lower_cofaces(
        objective: ObjectiveSpec,
        move: MoveSpec,
        sampler: SamplerSpec,
        context: _LatticeContextSpec,
        target: _LatticeStateSpec,
    ) -> LoweredObjective:
        """Return the table, classes, symbols, expression, and incidence of an objective.

        The objective is on cofaces.
        """
        owner = f'The Monte Carlo laws of {context.owner}'
        table, signs, boundary = LatticeProblemPass._get_coface_tables(target, owner)
        if sampler.schedule == 'checkerboard':
            error_msg = f"{owner} sum over cofaces, which need the 'colored' or 'random-sequential' schedule."
            raise ValidationError(error_msg)
        classes = LatticeProblemPass._get_coface_classes(boundary) if sampler.schedule == 'colored' else ()
        rests = [symbol(f'_neighbor_{index}') for index in range(table.shape[1])]
        orientations = [symbol(f'_sign_{index}') for index in range(table.shape[1])]
        masks = [symbol(f'_mask_{index}') for index in range(table.shape[1])]
        columns = (rests, orientations, masks)
        expression = LatticeProblemPass._lower_coface_sums(SR(objective.expression), move.target, columns, owner)
        return table, classes, [*rests, *orientations], masks, expression, (boundary, signs)

    @staticmethod
    def _compile_monte_carlo(
        objective: ObjectiveSpec,
        move: MoveSpec,
        sampler: SamplerSpec,
        context: _LatticeContextSpec,
    ) -> MonteCarloKernel:
        """Return Monte Carlo sampling compiled from an objective, a move, and a sampler.

        The objective is a local objective.

        Args:
            objective:
                The local objective.

            move:
                The move.

            sampler:
                The sampler.

            context:
                The lattice states and symbols of their component.

        Returns:
            sampling:
                The compiled sampling.
        """
        owner = f'The Monte Carlo laws of {context.owner}'
        target = LatticeProblemPass._get_lattice_state(context, move.target, 'A move')
        if sampler.kind == 'heat-bath' and move.kind in {'rotate', 'gaussian', 'function'}:
            error_msg = (
                f'{owner} use a heat-bath sampler with a {move.kind} move, which has no finite set of values to draw '
                'from.'
            )
            raise ValidationError(error_msg)
        cofaces = isinstance(objective.over, CofaceNeighborhood) or any(
            isinstance(item.over, CofaceNeighborhood) for item in neighborhoods_in(objective.expression)
        )
        if cofaces and move.kind == 'cluster':
            error_msg = f'{owner} make cluster moves on a sum over cofaces. Use a rotate or a random-value move.'
            raise ValidationError(error_msg)
        sampler = LatticeProblemPass._resolve_schedule(sampler, target.space, objective.over, cofaces)
        lower = LatticeProblemPass._lower_cofaces if cofaces else LatticeProblemPass._lower_neighbors
        table, classes, neighbours, masks, expression, faces = lower(objective, move, sampler, context, target)
        masked = any(item in masks for item in expression.variables())
        replacements = {}
        for call in LatticeProblemPass._get_calls(expression, 'neighbor'):
            state, index = call.operands()
            if state != move.target:
                error_msg = f'{owner} read the neighbours of {state}, but only the moved state {move.target} has them.'
                raise ValidationError(error_msg)
            replacements[call] = neighbours[int(index)]
        proposal = symbol('_proposal')
        local = expression.subs(replacements)
        change = local.subs({move.target: proposal}) - local
        bond = (
            LatticeProblemPass._get_bond_of(local, move.target, neighbours, masks, owner)
            if move.kind == 'cluster'
            else None
        )
        time = symbol('t')
        known = {move.target, proposal, time, *neighbours, *masks}
        extra = sorted(
            {item for item in (*change.variables(), *SR(sampler.intensity).variables()) if item not in known},
            key=str,
        )
        for item in extra:
            if item not in context.arguments:
                error_msg = f'{owner} read {item}, which is not a parameter or state of the lattice group.'
                raise ValidationError(error_msg)
        found = [context.arguments[item] for item in extra]
        arguments = _ArgumentsSpec(
            tuple(str(item) for item in extra),
            tuple(item[1] for item in found),
            tuple(item[2] for item in found),
            tuple(item[3] for item in found),
        )
        reads = [move.target, *neighbours, proposal, *(masks if masked else []), *extra]
        objective_change = ExpressionPrinter(time=time, compiled=True).prepare(reads).run([change]).code_
        intensity = ExpressionPrinter(time=time, compiled=True).prepare(extra).run([SR(sampler.intensity)]).code_
        if not callable(objective_change) or not callable(intensity):
            error_msg = 'The compiled printer returned no function of the objective change or of the intensity.'
            raise TypeError(error_msg)
        bond_change = (
            None
            if bond is None
            else ExpressionPrinter(time=time, compiled=True)
            .prepare([move.target, neighbours[0], *extra])
            .run([bond])
            .code_
        )
        if bond is not None and not callable(bond_change):
            error_msg = 'The compiled printer returned no function of the objective of a bond.'
            raise TypeError(error_msg)
        return MonteCarloKernel(
            context.owner,
            target,
            table,
            objective_change,
            arguments,
            intensity=intensity,
            move=move,
            sampler=sampler,
            classes=classes,
            masked=masked,
            bond=bond_change if callable(bond_change) else None,
            boundary=faces[0] if faces is not None else None,
            signs=faces[1] if faces is not None else None,
        ).prepare()

    @staticmethod
    def _schedule_of(law: object) -> str | None:
        """Return the schedule of a local rule or of the sampler of a sample.

        Without one, it is that of the default sampler.
        """
        if isinstance(law, _LocalRuleLaw):
            return law.schedule
        if isinstance(law, _SampleLaw):
            return (law.sampler or SamplerSpec()).schedule
        return None

    @staticmethod
    def _schedule_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the lattice laws whose checkerboard schedule needs a square lattice.

        Their space is not a square lattice.
        """
        records = Compilation._get_required(compilation.census, 'census').records
        issues = []
        for record in records:
            square = isinstance(record.space, LatticeSpace) and record.space.geometry == 'square'
            if not square and LatticeProblemPass._schedule_of(record.law) == 'checkerboard':
                message = (
                    f'The {record.law._get_name()} of {record.path} uses the checkerboard schedule, which needs a '
                    f'square lattice, but the component lives on {record.space!r}.'
                )
                fix = (
                    "Use the 'synchronous' or the 'random-sequential' schedule of a local rule, or the "
                    "'random-sequential' or the 'colored' schedule of a sampler."
                )
                issues.append(IssueSpec('schedule-unsupported', record.path, message, fix))
        return issues
