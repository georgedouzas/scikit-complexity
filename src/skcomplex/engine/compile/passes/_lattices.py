"""Compile the laws of lattices: local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast

import networkx as nx
import numba
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, DriveLaw, ExtremalLaw, LocalRuleLaw, RelaxationLaw, SampleLaw
from ....core.laws.utils import OuterTotalistic, Table, Totalistic, neighborhoods_in
from ....core.spaces import (
    AdjacencyNeighborhood,
    BaseSpace,
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    NetworkSpace,
)
from ....core.specs import EnergySpec, MoveSpec, SamplerSpec
from ...backends import resolve_backend
from ...problems import ExpressionPrinter, Problem, Source, symbol
from ...problems._arguments import Arguments, Indices
from ...problems._drive_kernel import DriveKernel
from ...problems._extremal_kernel import ExtremalKernel
from ...problems._lattice_state import LatticeState
from ...problems._lookup import Lookup, Vector
from ...problems._monte_carlo_kernel import MonteCarloKernel
from ...problems._relaxation_kernel import RelaxationKernel
from ...problems._rule_kernel import RuleKernel
from .._base import BasePass
from .._census import Census
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._issue import Issue
from .._kernel_spec import KernelSpec
from .._lattice_context import LatticeContext
from .._laws import LATTICE_LAWS

_COLORING_STRATEGY = 'largest_first'

type LatticeStep = RuleKernel | DriveKernel | RelaxationKernel | MonteCarloKernel | ExtremalKernel


class LatticeProblemPass(BasePass):
    """Pass that adds the steps of local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the lattice steps to the problem."""
        compilation = self.compilation_
        declarations = cast('Declarations', compilation.layout)
        members = Declarations._member_count(list(declarations.blocks.values()))
        fused = resolve_backend(compilation.backend, members).name == 'numba'
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            lattice=self._compile_lattice(cast('Laws', compilation.reduced), declarations, fused),
        )

    @staticmethod
    def _lattice_state(context: LatticeContext, item: Expression, law: str) -> LatticeState:
        """Return the lattice state of a target symbol, or raise an error naming the law."""
        if item not in context.states:
            error_msg = f'{law} of {context.owner} targets {item}, which is not a state of a lattice group.'
            raise ValidationError(error_msg)
        return context.states[item]

    @staticmethod
    def _keywords(context: LatticeContext, function: Callable[..., object]) -> Arguments:
        """Return the parameters a function accepts by keyword."""
        by_name = {
            name: (is_state, start, size) for name, is_state, start, size in context.arguments.values() if not is_state
        }
        names = LatticeProblemPass._accepted(function, list(by_name))
        return Arguments(
            names,
            tuple(by_name[name][0] for name in names),
            tuple(by_name[name][1] for name in names),
            tuple(by_name[name][2] for name in names),
        )

    @staticmethod
    def _accepted(function: Callable[..., object], names: Sequence[str]) -> tuple[str, ...]:
        """Return the keyword names a function accepts among some names."""
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return tuple(names)
        parameters = signature.parameters.values()
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters):
            return tuple(names)
        keywords = {item.name for item in parameters if item.kind in {item.KEYWORD_ONLY, item.POSITIONAL_OR_KEYWORD}}
        return tuple(name for name in names if name in keywords)

    @staticmethod
    def _lattice_context(spec: KernelSpec, declarations: Declarations, fused: bool) -> LatticeContext:
        """Return the lattice states, the arguments, and the neighbour symbols of the symbols a component reads."""
        states, arguments, others = {}, {}, {}
        for item, reference in spec.references.items():
            entry = reference.entry
            owner, _, variable = entry.name.rpartition('.')
            block = declarations.blocks.get(owner)
            space = block.node.space if block is not None else None
            is_lattice = (
                isinstance(space, LatticeSpace | NetworkSpace | CayleySpace)
                and block is not None
                and block.node.members is not None
                and not block.node.is_agent
            )
            if reference.is_state and is_lattice and not entry.member_shape:
                states[item] = LatticeState(entry.name, entry.start, space)  # type: ignore[arg-type]
            if not entry.member_shape:
                arguments[item] = (variable, reference.is_state, entry.start, entry.size)
            if reference.source is Source.OTHER:
                others[item] = entry.name
        return LatticeContext(spec.owner, states, arguments, fused, others)

    @staticmethod
    def _monte_carlo_step(spec: KernelSpec, law: SampleLaw, context: LatticeContext) -> LatticeStep:
        """Return the Monte Carlo sampling of a sample, with Metropolis acceptance at `temperature` by default."""
        sampler = law.sampler
        if sampler is None:
            if not hasattr(spec.namespace, 'temperature'):
                error_msg = f'{spec.owner} declares a sample without a sampler and no parameter `temperature`.'
                raise ValidationError(error_msg)
            sampler = SamplerSpec(temperature=spec.namespace.temperature)
        return LatticeProblemPass._compile_monte_carlo(law.energy, law.move, sampler, context)

    @staticmethod
    def _lattice_step(spec: KernelSpec, law: BaseLaw, context: LatticeContext) -> LatticeStep | None:
        """Return the step of a lattice law, or `None` for a law of another kind."""
        if isinstance(law, LocalRuleLaw):
            return LatticeProblemPass._compile_rule(law, context)
        if isinstance(law, DriveLaw):
            return LatticeProblemPass._compile_drive(law, context)
        if isinstance(law, RelaxationLaw):
            return LatticeProblemPass._compile_relaxation(law, context)
        if isinstance(law, ExtremalLaw):
            return LatticeProblemPass._compile_extremal(law, context)
        if isinstance(law, SampleLaw):
            return LatticeProblemPass._monte_carlo_step(spec, law, context)
        return None

    @staticmethod
    def _compile_lattice(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: Declarations,
        fused: bool,
    ) -> tuple[LatticeStep, ...]:
        """Return the steps of local rules, drives, relaxations, extremal laws, and samples in declaration order."""
        steps: list[LatticeStep] = []
        for spec, laws in specs:
            if not any(isinstance(law, LATTICE_LAWS) for law in laws):
                continue
            context = LatticeProblemPass._lattice_context(spec, declarations, fused)
            found = [LatticeProblemPass._lattice_step(spec, law, context) for law in laws]
            steps += [step for step in found if step is not None]
        return tuple(steps)

    @staticmethod
    def _as_table(rule: object) -> object:
        """Return a rule code as its table, through its `table()` method, and any other rule unchanged."""
        convert = getattr(rule, 'table', None)
        return convert() if not callable(rule) and callable(convert) else rule

    @staticmethod
    def _calls(expression: Expression, name: str) -> list[Expression]:
        """Return the calls of a symbolic function by name inside an expression, outermost first."""
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
    def _center_columns(neighborhood: LatticeNeighborhood | AdjacencyNeighborhood | MemoryNeighborhood) -> int:
        """Return the number of columns of a neighbourhood that read the site itself."""
        if isinstance(neighborhood, AdjacencyNeighborhood):
            return int(neighborhood.center)
        if isinstance(neighborhood, MemoryNeighborhood):
            return sum(element.strip() in {'', '1'} for element in neighborhood.check().prepare().elements_)
        return sum(not any(offset) for offset in neighborhood.check().prepare().offsets_)

    @staticmethod
    def _check_checkerboard(
        space: BaseSpace,
        neighborhood: LatticeNeighborhood | AdjacencyNeighborhood | MemoryNeighborhood,
        owner: str,
    ) -> LatticeSpace:
        """Return the square lattice of a checkerboard update, or raise an error when neighbours change together."""
        if (
            not isinstance(space, LatticeSpace)
            or space.geometry != 'square'
            or not isinstance(neighborhood, LatticeNeighborhood)
        ):
            error_msg = (
                f"{owner} uses a checkerboard schedule, which needs a square lattice. Use the 'colored' schedule, "
                'which updates the classes of a proper coloring of the neighbour graph in turn.'
            )
            raise ValidationError(error_msg)
        odd = all(sum(offset) % 2 for offset in neighborhood.check().prepare().offsets_)
        even = space.boundary != 'periodic' or all(size % 2 == 0 for size in space.shape_)
        if not (odd and even):
            error_msg = (
                f'{owner} uses a checkerboard schedule, which needs neighbours on the other sublattice and even sizes '
                "on periodic lattices. Use the 'random-sequential' schedule instead."
            )
            raise ValidationError(error_msg)
        return space

    @staticmethod
    def _color_classes(table: Indices) -> tuple[Indices, ...]:
        """Return the classes of a proper coloring of the graph that joins every site to the sites of its table row."""
        size = table.shape[0]
        rows = np.repeat(np.arange(size), table.shape[1])
        columns = table.ravel()
        real = (columns != size) & (columns != rows)
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(zip(rows[real].tolist(), columns[real].tolist(), strict=True))
        colors = nx.greedy_color(graph, strategy=_COLORING_STRATEGY)
        found = np.asarray([colors[site] for site in range(size)], dtype=np.intp)
        return tuple(np.flatnonzero(found == color) for color in range(int(found.max(initial=-1)) + 1))

    @staticmethod
    @functools.cache
    def _sublattices(space: LatticeSpace) -> tuple[Indices, Indices]:
        """Return the sites of even and of odd coordinate sum."""
        colors = np.indices(space.shape_).reshape(space.n_dim_, -1).sum(axis=0) % 2
        return np.flatnonzero(colors == 0), np.flatnonzero(colors == 1)

    @staticmethod
    def _schedule_classes(
        schedule: str,
        space: BaseSpace,
        neighborhood: LatticeNeighborhood | AdjacencyNeighborhood | MemoryNeighborhood,
        table: Indices,
        owner: str,
    ) -> tuple[Indices, ...]:
        """Return the classes of sites a checkerboard or colored schedule updates in turn, and none otherwise."""
        if schedule == 'checkerboard':
            return LatticeProblemPass._sublattices(LatticeProblemPass._check_checkerboard(space, neighborhood, owner))
        if schedule == 'colored':
            return LatticeProblemPass._color_classes(table)
        return ()

    @staticmethod
    def _compiles_in_numba(rule: Callable[..., Any]) -> bool:
        """Check whether a rule takes its arguments by position or keyword only, the form Numba compiles."""
        try:
            parameters = inspect.signature(rule).parameters.values()
        except (TypeError, ValueError):
            return False
        return all(item.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for item in parameters)

    @staticmethod
    def _degrees_of(
        table: Indices,
        neighborhood: LatticeNeighborhood | AdjacencyNeighborhood | MemoryNeighborhood,
    ) -> Vector:
        """Return the number of real neighbours of every site, besides the site itself."""
        return ((table != table.shape[0]).sum(axis=1) - LatticeProblemPass._center_columns(neighborhood)).astype(float)

    @staticmethod
    def _extremal_source(
        targets: tuple[LatticeState, ...], key: LatticeState, maximum: bool, arguments: Arguments,
    ) -> str:
        """Return the source of the compiled loop of the applications of an extremal law."""
        values = ', '.join(f'x[{state.start} + sites]' for state in targets)
        keywords = ''.join(
            f', {name}=p[{start}]' if size == 1 else f', {name}=p[{start}:{start + size}][sites]'
            for name, start, size in zip(arguments.names, arguments.starts, arguments.sizes, strict=True)
        )
        if len(targets) == 1:
            assign = f'        x[{targets[0].start} + sites] = result\n'
        else:
            assign = ''.join(
                f'        x[{state.start} + sites] = result[{index}]\n' for index, state in enumerate(targets)
            )
        return (
            'def loop(x, p, rng, members, counts, repeat, sites_out, values_out):\n'
            '    for application in range(repeat):\n'
            f'        keys = x[{key.start}:{key.start + key.size}]\n'
            f'        best = keys.{"max" if maximum else "min"}()\n'
            '        ties = np.flatnonzero(keys == best)\n'
            '        site = ties[rng.integers(0, ties.size)] if ties.size > 1 else ties[0]\n'
            '        sites = members[site, : counts[site]]\n'
            '        random = rng.random(sites.size)\n'
            f'        result = rule({values}, random=random{keywords})\n'
            f'{assign}'
            '        sites_out[application] = site\n'
            '        values_out[application] = best\n'
        )

    @staticmethod
    def _extremal_loop(
        rule: Callable[..., Any],
        targets: tuple[LatticeState, ...],
        key: LatticeState,
        maximum: bool,
        arguments: Arguments,
    ) -> Callable[..., None] | None:
        """Return the compiled loop of the applications of an extremal law, or `None` when Numba cannot compile it."""
        if not LatticeProblemPass._compiles_in_numba(rule):
            return None
        namespace: dict[str, Any] = {'np': np, 'rule': numba.njit(error_model='numpy')(rule)}
        source = LatticeProblemPass._extremal_source(targets, key, maximum, arguments)
        exec(source, namespace)  # noqa: S102  # nosec B102: runs the loop source generated above
        loop = numba.njit(namespace['loop'])
        end = max(state.start + state.size for state in (*targets, key))
        width = max((start + size for start, size in zip(arguments.starts, arguments.sizes, strict=True)), default=1)
        try:
            loop(
                np.zeros(end),
                np.zeros(width),
                np.random.default_rng(0),
                np.zeros((1, 1), dtype=np.intp),
                np.zeros(1, dtype=np.intp),
                0,
                np.zeros(0, dtype=np.intp),
                np.zeros(0),
            )
        except numba.core.errors.NumbaError:
            return None
        return loop  # type: ignore[no-any-return, unused-ignore]

    @staticmethod
    def _extremal_members(table: Indices) -> tuple[Indices, Indices]:
        """Return every site followed by its real neighbours without duplicates, padded with -1, and their counts."""
        size, width = table.shape
        members = np.full((size, width + 1), -1, dtype=np.intp)
        counts = np.zeros(size, dtype=np.intp)
        for site in range(size):
            unique = list(dict.fromkeys([site, *(int(item) for item in table[site] if item != size)]))
            members[site, : len(unique)] = unique
            counts[site] = len(unique)
        return members, counts

    @staticmethod
    def _index_table(
        space: BaseSpace,
        neighborhood: LatticeNeighborhood | AdjacencyNeighborhood | MemoryNeighborhood,
        owner: str,
    ) -> Indices:
        """Return the neighbour table of a lattice or a network, or raise an error naming the law that reads it."""
        try:
            return space._index_table(neighborhood)
        except (ValueError, ValidationError) as error:
            error_msg = f'{owner}: {error}'
            raise ValidationError(error_msg) from error

    @staticmethod
    def _lookup_of(rule: object, n_offsets: int, owner: str) -> Lookup | None:
        """Return the lookup array of a table or totalistic rule, or `None` for a function rule."""
        if isinstance(rule, Table):
            base = rule.n_states
            values = np.full(base**n_offsets, float(rule.default))
            for pattern, value in rule.mapping.items():
                if len(pattern) != n_offsets:
                    error_msg = f'{owner} has a table pattern {pattern} for a neighbourhood of {n_offsets} cells.'
                    raise ValidationError(error_msg)
                values[int(np.ravel_multi_index(tuple(int(item) for item in pattern), (base,) * n_offsets))] = value
            return Lookup('table', base, values)
        if isinstance(rule, Totalistic | OuterTotalistic):
            sums = np.arange(n_offsets * (rule.states - 1) + 1)
            if isinstance(rule, Totalistic):
                if callable(rule.rule):
                    values = np.broadcast_to(np.asarray(rule.rule(sums), dtype=float), sums.shape).copy()
                else:
                    values = np.asarray([float(rule.rule.get(int(total), rule.default)) for total in sums])
                return Lookup('totalistic', rule.states, values)
            centers = np.arange(rule.states)[:, None]
            if callable(rule.rule):
                grid = np.broadcast_to(
                    np.asarray(rule.rule(centers, sums[None, :]), dtype=float), (rule.states, sums.size),
                )
            else:
                grid = np.asarray(
                    [
                        [float(rule.rule.get((int(center), int(total)), rule.default)) for total in sums]
                        for center in range(rule.states)
                    ],
                )
            return Lookup('outer', rule.states, grid.ravel().copy(), sums.size)
        return None

    @staticmethod
    def _lower_neighbor_sums(  # noqa: PLR0917
        expression: Expression,
        target: Expression,
        energy: EnergySpec,
        context: LatticeContext,
        neighbors: Sequence[Expression],
        masks: Sequence[Expression],
    ) -> Expression:
        """Return an expression with every neighbour sum written as the masked sum over the columns of the table."""
        owner = f'The Monte Carlo laws of {context.owner}'
        moved = context.states[target].name
        replacements = {}
        for reduction in neighborhoods_in(expression):
            if reduction.kind != 'sum' or reduction.over not in (None, energy.over):
                error_msg = (
                    f'{owner} reduce {reduction.call}, but a local energy only sums over the neighbourhood of the '
                    f'energy, {energy.over!r}.'
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
                (mask * column for mask, column in zip(masks, columns, strict=True)), SR(0),
            )
        return SR(expression).subs(replacements) if replacements else SR(expression)

    @staticmethod
    def _compile_drive(law: DriveLaw, context: LatticeContext) -> DriveKernel:
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
        return DriveKernel(
            context.owner,
            LatticeProblemPass._lattice_state(context, law._get_targets()[0], 'A drive'),
            law.function,
            LatticeProblemPass._keywords(context, law.function),
        ).check()

    @staticmethod
    def _compile_rule(law: LocalRuleLaw, context: LatticeContext) -> RuleKernel:
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
            LatticeProblemPass._lattice_state(context, item, 'A local rule') for item in law._get_targets()
        )
        reads = tuple(
            LatticeProblemPass._lattice_state(context, item, 'A local rule')
            for item in (law.reads or law._get_targets())
        )
        spaces = {id(state.space) for state in (*targets, *reads)}
        if len(spaces) > 1:
            error_msg = f'{owner} reads and updates states of different lattices.'
            raise ValidationError(error_msg)
        space = targets[0].space
        table = LatticeProblemPass._index_table(space, law.over, owner)
        rule = LatticeProblemPass._as_table(law.rule)
        if isinstance(space, NetworkSpace) and isinstance(rule, Table):
            error_msg = (
                f'{owner} uses a table of ordered patterns on a network, whose sites have different degrees. Use a '
                'totalistic or an outer totalistic rule, or a function rule.'
            )
            raise ValidationError(error_msg)
        lookup = LatticeProblemPass._lookup_of(rule, table.shape[1], owner)
        if lookup is not None and (len(targets) > 1 or len(reads) > 1):
            error_msg = f'{owner} uses a lookup rule, which updates and reads a single state.'
            raise ValidationError(error_msg)
        if lookup is None and not callable(rule):
            error_msg = (
                f'{owner} has the rule {law.rule!r}, which is not a function, a Table, a totalistic rule, or a rule '
                'code with a table() method.'
            )
            raise ValidationError(error_msg)
        reads_degree = lookup is None and 'degree' in LatticeProblemPass._accepted(law.rule, ['degree'])  # type: ignore[arg-type]
        return RuleKernel(
            context.owner,
            targets,
            reads,
            table,
            None if lookup is not None else law.rule,  # type: ignore[arg-type]
            lookup,
            law.schedule,
            law.stochastic,
            LatticeProblemPass._keywords(context, law.rule) if lookup is None else Arguments(),  # type: ignore[arg-type]
            context.fused and lookup is not None and law.schedule == 'synchronous',
            LatticeProblemPass._degrees_of(table, law.over) if reads_degree else None,
            LatticeProblemPass._schedule_classes(law.schedule, space, law.over, table, owner),
        ).check()

    @staticmethod
    def _compile_part(law: object, context: LatticeContext) -> RuleKernel | DriveKernel:
        """Return a law of a relaxation compiled."""
        if isinstance(law, LocalRuleLaw):
            return LatticeProblemPass._compile_rule(law, context)
        if isinstance(law, DriveLaw):
            return LatticeProblemPass._compile_drive(law, context)
        error_msg = f'A relaxation of {context.owner} holds {law!r}, which is not a local rule or a drive.'
        raise ValidationError(error_msg)

    @staticmethod
    def _compile_relaxation(law: RelaxationLaw, context: LatticeContext) -> RelaxationKernel:
        """Return a relaxation compiled for its lattice.

        Args:
            law:
                The relaxation.

            context:
                The lattice states and symbols of its component.

        Returns:
            relaxation:
                The compiled relaxation.
        """
        if not law.relax:
            error_msg = f'A relaxation of {context.owner} has no relaxation laws.'
            raise ValidationError(error_msg)
        return RelaxationKernel(
            law.label or context.owner,
            tuple(LatticeProblemPass._compile_part(item, context) for item in law.drive),
            tuple(LatticeProblemPass._compile_part(item, context) for item in law.relax),
            tuple(LatticeProblemPass._lattice_state(context, item, 'A relaxation') for item in law._get_targets()),
            law.stable,
            int(law.max_iterations),
            law.activity,
        ).check()

    @staticmethod
    def _compile_extremal(law: ExtremalLaw, context: LatticeContext) -> ExtremalKernel:
        """Return an extremal law compiled for its lattice or network.

        Args:
            law:
                The extremal law.

            context:
                The lattice states and symbols of its component.

        Returns:
            extremal:
                The compiled law, with a compiled loop on the Numba backend when Numba compiles the rule.

        Raises:
            ValidationError:
                The targets and the key are not states of one lattice, or the neighbourhood does not fit it.
        """
        owner = f'The extremal law of {context.owner}'
        targets = tuple(
            LatticeProblemPass._lattice_state(context, item, 'An extremal law') for item in law._get_targets()
        )
        key = LatticeProblemPass._lattice_state(context, law._get_key(), 'An extremal law')
        if len({id(state.space) for state in (*targets, key)}) > 1:
            error_msg = f'{owner} reads and updates states of different lattices.'
            raise ValidationError(error_msg)
        members, counts = LatticeProblemPass._extremal_members(
            LatticeProblemPass._index_table(key.space, law.over, owner),
        )
        arguments = LatticeProblemPass._keywords(context, law.rule)
        maximum = law.extremum == 'max'
        loop = LatticeProblemPass._extremal_loop(law.rule, targets, key, maximum, arguments) if context.fused else None
        return ExtremalKernel(
            law.label or context.owner,
            context.owner,
            targets,
            key,
            members,
            counts,
            law.rule,
            arguments,
            maximum,
            int(law.repeat),
            loop,
        ).check()

    @staticmethod
    def _compile_monte_carlo(
        energy: EnergySpec,
        move: MoveSpec,
        sampler: SamplerSpec,
        context: LatticeContext,
    ) -> MonteCarloKernel:
        """Return Monte Carlo sampling compiled from a local energy, a move, and a sampler.

        Args:
            energy:
                The local energy.

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
        target = LatticeProblemPass._lattice_state(context, move.target, 'A move')
        if sampler.kind == 'heat-bath' and move.kind == 'rotate':
            error_msg = (
                f'{owner} use a heat-bath sampler with a rotation, which has no finite set of values to draw from.'
            )
            raise ValidationError(error_msg)
        space = target.space
        table = LatticeProblemPass._index_table(space, energy.over, owner)
        classes = LatticeProblemPass._schedule_classes(sampler.schedule, space, energy.over, table, owner)
        neighbours = [symbol(f'_neighbor_{index}') for index in range(table.shape[1])]
        masks = [symbol(f'_mask_{index}') for index in range(table.shape[1])]
        expression = LatticeProblemPass._lower_neighbor_sums(
            SR(energy.expression), move.target, energy, context, neighbours, masks,
        )
        masked = any(item in masks for item in expression.variables())
        replacements = {}
        for call in LatticeProblemPass._calls(expression, 'neighbor'):
            state, index = call.operands()
            if state != move.target:
                error_msg = f'{owner} read the neighbours of {state}, but only the moved state {move.target} has them.'
                raise ValidationError(error_msg)
            replacements[call] = neighbours[int(index)]
        proposal = symbol('_proposal')
        local = expression.subs(replacements)
        change = local.subs({move.target: proposal}) - local
        known = {move.target, proposal, *neighbours, *masks}
        extra = sorted(
            {item for item in (*change.variables(), *SR(sampler.temperature).variables()) if item not in known},
            key=str,
        )
        for item in extra:
            if item not in context.arguments:
                error_msg = f'{owner} read {item}, which is not a parameter or state of the lattice group.'
                raise ValidationError(error_msg)
        found = [context.arguments[item] for item in extra]
        arguments = Arguments(
            tuple(str(item) for item in extra),
            tuple(item[1] for item in found),
            tuple(item[2] for item in found),
            tuple(item[3] for item in found),
        )
        time = symbol('_time')
        reads = [move.target, *neighbours, proposal, *(masks if masked else []), *extra]
        return MonteCarloKernel(
            context.owner,
            target,
            table,
            ExpressionPrinter(time=time, compiled=True).check().prepare(reads).run([change]),
            arguments,
            ExpressionPrinter(time=time, compiled=True).check().prepare(extra).run([SR(sampler.temperature)]),
            move,
            sampler,
            classes,
            masked,
        ).check()

    @staticmethod
    def _schedule_of(law: object) -> str | None:
        """Return the schedule of a local rule or of the sampler of a sample, the default sampler's without one."""
        if isinstance(law, LocalRuleLaw):
            return law.schedule
        if isinstance(law, SampleLaw):
            return (law.sampler or SamplerSpec()).schedule
        return None

    @staticmethod
    def _schedule_issues(compilation: Compilation) -> list[Issue]:
        """Return the lattice laws whose checkerboard schedule needs a square lattice that their space is not."""
        records = cast('Census', compilation.census).records
        issues = []
        for record in records:
            square = isinstance(record.space, LatticeSpace) and record.space.geometry == 'square'
            if not square and LatticeProblemPass._schedule_of(record.law) == 'checkerboard':
                message = (
                    f'The {type(record.law).__name__} of {record.path} uses the checkerboard schedule, which needs a '
                    f'square lattice, but the component lives on {record.space!r}.'
                )
                fix = (
                    "Use the 'synchronous' or the 'random-sequential' schedule of a local rule, or the "
                    "'random-sequential' or the 'colored' schedule of a sampler."
                )
                issues.append(Issue('schedule-unsupported', record.path, message, fix))
        return issues
