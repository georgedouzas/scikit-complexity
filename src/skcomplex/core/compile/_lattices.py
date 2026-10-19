"""Compile local rules, drives, relaxations, and Monte Carlo laws into steps over lattice states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, SolverError, ValidationError, symbol
from ..model import (
    Drive,
    Energy,
    LatticeSpace,
    LocalRule,
    Move,
    Neighborhood,
    OuterTotalistic,
    Relaxation,
    Sampler,
    Table,
    Totalistic,
    Wolfram,
)
from ._printing import numpy_function

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]

LOOKUP_KINDS = {'table': 0, 'totalistic': 1, 'outer': 2}


@dataclass(frozen=True)
class RelaxationRecord:
    """Relaxation of a lattice within one step, such as an avalanche.

    Args:
        label:
            The label of the relaxation.

        time:
            The time of the step.

        size:
            The number of site updates.

        duration:
            The number of relaxation iterations.

        area:
            The number of distinct sites that changed.
    """

    label: str
    time: float
    size: int
    duration: int
    area: int


@dataclass(frozen=True)
class LatticeState:
    """State of a lattice group in the state vector.

    Args:
        name:
            The dotted name of the state.

        start:
            The index of its first component.

        space:
            The lattice of the group.
    """

    name: str
    start: int
    space: LatticeSpace
    shape: tuple[int, ...] = field(init=False, repr=False)
    size: int = field(init=False, repr=False)
    sites: slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape', self.space.dimensions)
        object.__setattr__(self, 'size', self.space.size)
        object.__setattr__(self, 'sites', slice(self.start, self.start + self.size))


@dataclass(frozen=True)
class Arguments:
    """Parameters and states that a lattice law reads besides its neighbourhoods, by name.

    Args:
        names:
            The keyword or symbol names.

        vectors:
            Whether every argument is read from the state vector rather than the parameter vector.

        starts:
            The index of the first component of every argument.

        sizes:
            The number of components of every argument, the number of sites for per-site values or 1 for shared ones.
    """

    names: tuple[str, ...] = ()
    vectors: tuple[bool, ...] = ()
    starts: tuple[int, ...] = ()
    sizes: tuple[int, ...] = ()

    def values(self, x: Vector, p: Vector, sites: Indices | None = None) -> dict[str, Any]:
        """Return the values of the arguments, per-site values restricted to some sites.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            sites:
                The sites to keep, or `None` for all.

        Returns:
            values:
                The value of every argument by name.
        """
        found: dict[str, Any] = {}
        for name, is_state, start, size in zip(self.names, self.vectors, self.starts, self.sizes, strict=True):
            source = x if is_state else p
            if size == 1:
                found[name] = float(source[start])
                continue
            values = source[start : start + size]
            found[name] = values if sites is None else values[sites]
        return found


def _accepted(function: Callable[..., Any], names: Sequence[str]) -> tuple[str, ...]:
    """Return the keyword names a function accepts among some names."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return tuple(names)
    parameters = signature.parameters.values()
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters):
        return tuple(names)
    keywords = {item.name for item in parameters if item.kind in {item.KEYWORD_ONLY, item.POSITIONAL_OR_KEYWORD}}
    return tuple(name for name in names if name in keywords)


@functools.cache
def _sublattices(space: LatticeSpace) -> tuple[Indices, Indices]:
    """Return the sites of even and of odd coordinate sum."""
    colors = np.indices(space.dimensions).reshape(space.n_dim, -1).sum(axis=0) % 2
    return np.flatnonzero(colors == 0), np.flatnonzero(colors == 1)


def _check_checkerboard(space: LatticeSpace, neighborhood: Neighborhood, owner: str) -> None:
    """Raise an error when a checkerboard update would let neighbours of one sublattice change together."""
    odd = all(sum(offset) % 2 for offset in neighborhood.offsets)
    even = space.boundary != 'periodic' or all(size % 2 == 0 for size in space.dimensions)
    if not (odd and even):
        error_msg = (
            f'{owner} uses a checkerboard schedule, which needs neighbours on the other sublattice and even sizes on '
            "periodic lattices. Use the 'random-sequential' schedule instead."
        )
        raise ValidationError(error_msg)


@numba.njit(cache=True)
def _fused_lookup(  # noqa: PLR0917
    values: Vector,
    centers: Vector,
    fill: float,
    table: Indices,
    kind: int,
    base: int,
    lookup: Vector,
    columns: int,
) -> Vector:
    """Return the next values of a lookup rule at every site in one compiled loop."""
    n_sites, n_offsets = table.shape
    following = np.empty(n_sites)
    for site in range(n_sites):
        code = 0
        total = 0.0
        for offset in range(n_offsets):
            neighbour = table[site, offset]
            value = values[neighbour] if neighbour < n_sites else fill
            code = code * base + int(value)
            total += value
        if kind == 0:
            following[site] = lookup[code]
        elif kind == 1:
            following[site] = lookup[int(total)]
        else:
            following[site] = lookup[int(centers[site]) * columns + int(total)]
    return following


@dataclass(frozen=True)
class Lookup:
    """Lookup array of a table or totalistic rule.

    Args:
        kind:
            `'table'`, `'totalistic'`, or `'outer'`.

        base:
            The number of state values.

        values:
            The next value by pattern code, by sum, or by cell value and sum in row-major order.

        columns:
            The number of sums of an outer totalistic rule.
    """

    kind: str
    base: int
    values: Vector
    columns: int = 1


def _lookup_of(rule: object, n_offsets: int, owner: str) -> Lookup | None:
    """Return the lookup array of a table, Wolfram, or totalistic rule, or `None` for a function rule."""
    if isinstance(rule, Wolfram):
        if n_offsets != 3:  # noqa: PLR2004
            error_msg = f'{owner} uses a Wolfram rule, which reads 3 cells, with a neighbourhood of {n_offsets}.'
            raise ValidationError(error_msg)
        rule = rule.table()
    if isinstance(rule, Table):
        base = int(rule.states)  # type: ignore[arg-type]
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
            grid = np.broadcast_to(np.asarray(rule.rule(centers, sums[None, :]), dtype=float), (rule.states, sums.size))
        else:
            grid = np.asarray(
                [
                    [float(rule.rule.get((int(center), int(total)), rule.default)) for total in sums]
                    for center in range(rule.states)
                ],
            )
        return Lookup('outer', rule.states, grid.ravel().copy(), sums.size)
    return None


@dataclass(frozen=True)
class CompiledRule:
    """Local rule compiled for a lattice, with its neighbour table and its evaluation.

    Args:
        owner:
            The path of the component that declares the rule.

        targets:
            The lattice states the rule updates.

        reads:
            The lattice states whose neighbourhoods the rule reads.

        table:
            The member index of every neighbour of every site, with the number of sites for the fill value.

        rule:
            The function rule, or `None` for a lookup rule.

        lookup:
            The lookup array of a table or totalistic rule, or `None` for a function rule.

        schedule:
            The update schedule.

        stochastic:
            Whether the rule receives uniform random numbers.

        arguments:
            The parameters a function rule receives by keyword.

        fused:
            Whether a synchronous lookup rule runs as one compiled loop.
    """

    owner: str
    targets: tuple[LatticeState, ...]
    reads: tuple[LatticeState, ...]
    table: Indices
    rule: Callable[..., Any] | None
    lookup: Lookup | None
    schedule: str
    stochastic: bool
    arguments: Arguments = field(default_factory=Arguments)
    fused: bool = False

    @property
    def space(self) -> LatticeSpace:
        """Lattice of the rule."""
        return self.targets[0].space

    def _next(self, x: Vector, p: Vector, sites: Indices | None, rng: np.random.Generator) -> list[Vector]:
        """Return the next values of the targets at some sites, read from the current state."""
        space = self.space
        table = self.table if sites is None else self.table[sites]
        if self.lookup is not None:
            values = x[self.reads[0].sites]
            if self.fused and sites is None:
                centers = x[self.targets[0].sites]
                lookup = self.lookup
                following = _fused_lookup(
                    np.ascontiguousarray(values),
                    np.ascontiguousarray(centers),
                    float(space.fill),
                    table,
                    LOOKUP_KINDS[lookup.kind],
                    lookup.base,
                    lookup.values,
                    lookup.columns,
                )
                return [following]
            gathered = np.append(values, space.fill)[table].astype(np.intp)
            lookup = self.lookup
            if lookup.kind == 'table':
                codes = np.ravel_multi_index(tuple(gathered.T), (lookup.base,) * gathered.shape[1])
                return [lookup.values[codes]]
            totals = gathered.sum(axis=1)
            if lookup.kind == 'totalistic':
                return [lookup.values[totals]]
            centers = x[self.targets[0].sites].astype(np.intp)
            return [lookup.values[(centers if sites is None else centers[sites]) * lookup.columns + totals]]
        columns = [column for state in self.reads for column in np.append(x[state.sites], space.fill)[table].T]
        keywords = self.arguments.values(x, p, sites)
        if self.stochastic:
            keywords['random'] = rng.random(table.shape[0])
        result = self.rule(*columns, **keywords)  # type: ignore[misc]
        results = list(result) if len(self.targets) > 1 else [result]
        if len(results) != len(self.targets):
            error_msg = (
                f'The local rule of {self.owner} returned {len(results)} arrays for {len(self.targets)} targets.'
            )
            raise SolverError(error_msg)
        return [np.broadcast_to(np.asarray(item, dtype=float), (table.shape[0],)) for item in results]

    def _assign(self, x: Vector, values: list[Vector], sites: Indices | None) -> None:
        """Write the next values of the targets at some sites."""
        for state, value in zip(self.targets, values, strict=True):
            if sites is None:
                x[state.sites] = value
            else:
                x[state.start + sites] = value

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator) -> None:
        """Update the state vector in place by one application of the rule on its schedule.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.
        """
        size = self.targets[0].size
        if self.schedule == 'synchronous':
            self._assign(x, self._next(x, p, None, rng), None)
            return
        if self.schedule == 'checkerboard':
            for sites in _sublattices(self.space):
                self._assign(x, self._next(x, p, sites, rng), sites)
            return
        order = rng.permutation(size) if self.schedule == 'random-sequential' else rng.integers(size, size=size)
        for site in order:
            sites = np.asarray([site])
            self._assign(x, self._next(x, p, sites, rng), sites)


@dataclass(frozen=True)
class CompiledDrive:
    """Drive compiled for a lattice state.

    Args:
        owner:
            The path of the component that declares the drive.

        target:
            The lattice state the drive changes.

        function:
            The function of the state's array in lattice shape.

        arguments:
            The parameters the function receives by keyword.
    """

    owner: str
    target: LatticeState
    function: Callable[..., Any]
    arguments: Arguments = field(default_factory=Arguments)

    @property
    def targets(self) -> tuple[LatticeState, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator) -> None:
        """Update the state vector in place by one application of the drive.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.
        """
        shape = self.target.shape
        keywords = {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.values(x, p).items()
        }
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, random=rng, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()


@dataclass(frozen=True)
class CompiledRelaxation:
    """Relaxation compiled for a lattice, with its drive and relaxation laws.

    Args:
        label:
            The label of its events.

        drive:
            The laws applied once per step.

        relax:
            The laws applied until the lattice is stable.

        targets:
            The states the stability condition reads.

        stable:
            The stability condition on the targets' arrays in lattice shape.

        max_iterations:
            The largest number of relaxation iterations in one step.

        activity:
            The function that marks the active sites of an iteration, or `None` to count the sites that change.
    """

    label: str
    drive: tuple[CompiledRule | CompiledDrive, ...]
    relax: tuple[CompiledRule | CompiledDrive, ...]
    targets: tuple[LatticeState, ...]
    stable: Callable[..., bool]
    max_iterations: int
    activity: Callable[..., Any] | None = None

    def _arrays(self, x: Vector) -> list[Vector]:
        """Return the targets' arrays in lattice shape."""
        return [np.reshape(x[state.sites], state.shape) for state in self.targets]

    def _is_stable(self, x: Vector) -> bool:
        """Check the stability condition on the current state."""
        return bool(self.stable(*self._arrays(x)))

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator, time: float) -> RelaxationRecord | None:
        """Apply the drive, then relax until stable, and return the record of a relaxation that changed the lattice.

        Args:
            x:
                The state vector, updated in place.

            p:
                The parameter vector.

            rng:
                The random number generator.

            time:
                The time of the step.

        Returns:
            record:
                The size, duration, and area of the relaxation, or `None` when the lattice was already stable.

        Raises:
            SolverError:
                The lattice is not stable after the largest number of iterations.
        """
        for law in self.drive:
            law.apply(x, p, rng)
        size = duration = 0
        changed: npt.NDArray[np.bool_] | None = None
        sites = np.concatenate([np.arange(state.start, state.start + state.size) for state in self.targets])
        while not self._is_stable(x):
            if duration >= self.max_iterations:
                error_msg = (
                    f'The relaxation {self.label} is not stable after {self.max_iterations} iterations. Check its '
                    'stability condition or raise max_iterations.'
                )
                raise SolverError(error_msg)
            before = x[sites].copy()
            active = (
                np.ravel(np.asarray(self.activity(*self._arrays(x)), dtype=bool)) if self.activity is not None else None
            )
            for law in self.relax:
                law.apply(x, p, rng)
            updated = (x[sites] != before).reshape(len(self.targets), -1).any(axis=0) if active is None else active
            size += int(updated.sum())
            changed = updated if changed is None else changed | updated
            duration += 1
        if not duration:
            return None
        area = int(changed.sum()) if changed is not None else 0
        return RelaxationRecord(self.label, time, size, duration, area)


@dataclass(frozen=True)
class CompiledMonteCarlo:
    """Monte Carlo sampling of a lattice state compiled from its local energy, move, and sampler.

    Args:
        owner:
            The path of the component that declares the energy.

        target:
            The lattice state the move changes.

        table:
            The member index of every neighbour of every site.

        delta:
            The energy change `delta(value, *neighbours, proposal, **arguments)` over arrays of sites.

        arguments:
            The parameters and states the energy and the temperature read.

        temperature:
            The temperature as a function of the arguments.

        move:
            The move.

        sampler:
            The acceptance rule and the schedule.
    """

    owner: str
    target: LatticeState
    table: Indices
    delta: Callable[..., Any]
    arguments: Arguments
    temperature: Callable[..., Any]
    move: Move
    sampler: Sampler

    def _energy_change(
        self,
        x: Vector,
        values: Vector,
        sites: Indices,
        proposal: Vector,
        keywords: dict[str, Any],
    ) -> Vector:
        """Return the energy change of moving some sites to proposed values."""
        neighbours = np.append(x[self.target.sites], self.target.space.fill)[self.table[sites]]
        change = self.delta(0.0, values, *neighbours.T, proposal, *keywords.values())[0]
        return np.broadcast_to(np.asarray(change, dtype=float), values.shape)

    def _propose(self, values: Vector, rng: np.random.Generator) -> Vector:
        """Return proposed values for some sites."""
        if self.move.kind == 'flip':
            return -values
        if self.move.kind == 'random-value':
            return np.asarray(self.move.values)[rng.integers(len(self.move.values), size=values.size)]
        return values + rng.uniform(-self.move.width, self.move.width, size=values.size)

    def _update(self, x: Vector, p: Vector, sites: Indices, rng: np.random.Generator) -> None:
        """Propose and accept moves at sites that do not neighbour each other."""
        keywords = self.arguments.values(x, p, sites)
        temperature = np.asarray(self.temperature(0.0, *keywords.values())[0], dtype=float)
        if np.any(temperature <= 0):
            error_msg = f'The temperature of {self.owner} is {temperature}, but it must be positive.'
            raise SolverError(error_msg)
        values = x[self.target.start + sites]
        if self.sampler.kind == 'heat-bath':
            candidates = (
                np.unique(np.concatenate([values, -values]))
                if self.move.kind == 'flip'
                else np.asarray(self.move.values)
            )
            energies = np.stack(
                [self._energy_change(x, values, sites, np.full(values.shape, value), keywords) for value in candidates],
            )
            weights = np.exp(-(energies - energies.min(axis=0)) / temperature)
            cumulative = np.cumsum(weights / weights.sum(axis=0), axis=0)
            chosen = (rng.random(values.size) > cumulative).sum(axis=0)
            x[self.target.start + sites] = candidates[np.minimum(chosen, candidates.size - 1)]
            return
        proposal = self._propose(values, rng)
        change = self._energy_change(x, values, sites, proposal, keywords)
        with np.errstate(over='ignore'):
            if self.sampler.kind == 'metropolis':
                probability = np.exp(-np.maximum(change, 0.0) / temperature)
            else:
                probability = 1.0 / (1.0 + np.exp(change / temperature))
        accepted = rng.random(values.size) < probability
        x[self.target.start + sites[accepted]] = proposal[accepted]

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator) -> None:
        """Update the state vector in place by one sweep of the lattice.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.
        """
        size = self.target.size
        if self.sampler.schedule == 'checkerboard':
            for sites in _sublattices(self.target.space):
                self._update(x, p, sites, rng)
            return
        for site in rng.permutation(size):
            self._update(x, p, np.asarray([site]), rng)


type LatticeStep = CompiledRule | CompiledDrive | CompiledRelaxation | CompiledMonteCarlo


@dataclass
class LatticeContext:
    """What lattice laws are compiled against: the lattice of every state and the symbols a component reads.

    Args:
        owner:
            The path of the component that declares the laws.

        states:
            The lattice state of every target symbol.

        arguments:
            The parameters and states the component reads, by symbol, with their keyword names.

        fused:
            Whether lookup rules run as compiled loops.
    """

    owner: str
    states: Mapping[Expression, LatticeState]
    arguments: Mapping[Expression, tuple[str, bool, int, int]]
    fused: bool = False

    def state(self, item: Expression, law: str) -> LatticeState:
        """Return the lattice state of a target symbol, or raise an error naming the law."""
        if item not in self.states:
            error_msg = f'{law} of {self.owner} targets {item}, which is not a state of a lattice group.'
            raise ValidationError(error_msg)
        return self.states[item]

    def keywords(self, function: Callable[..., Any]) -> Arguments:
        """Return the parameters a function accepts by keyword."""
        by_name = {
            name: (is_state, start, size) for name, is_state, start, size in self.arguments.values() if not is_state
        }
        names = _accepted(function, list(by_name))
        return Arguments(
            names,
            tuple(by_name[name][0] for name in names),
            tuple(by_name[name][1] for name in names),
            tuple(by_name[name][2] for name in names),
        )


def compile_rule(law: LocalRule, context: LatticeContext) -> CompiledRule:
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
    targets = tuple(context.state(item, 'A local rule') for item in law.targets)
    reads = tuple(context.state(item, 'A local rule') for item in (law.reads or law.targets))
    spaces = {id(state.space) for state in (*targets, *reads)}
    if len(spaces) > 1:
        error_msg = f'{owner} reads and updates states of different lattices.'
        raise ValidationError(error_msg)
    space = targets[0].space
    try:
        table = space.index_table(law.neighborhood)
    except ValueError as error:
        error_msg = f'{owner}: {error}'
        raise ValidationError(error_msg) from error
    if law.schedule == 'checkerboard':
        _check_checkerboard(space, law.neighborhood, owner)
    lookup = _lookup_of(law.rule, len(law.neighborhood), owner)
    if lookup is not None and (len(targets) > 1 or len(reads) > 1):
        error_msg = f'{owner} uses a lookup rule, which updates and reads a single state.'
        raise ValidationError(error_msg)
    if lookup is None and not callable(law.rule):
        error_msg = f'{owner} has the rule {law.rule!r}, which is not a function, Table, Wolfram, or totalistic rule.'
        raise ValidationError(error_msg)
    return CompiledRule(
        context.owner,
        targets,
        reads,
        table,
        None if lookup is not None else law.rule,  # type: ignore[arg-type]
        lookup,
        law.schedule,
        law.stochastic,
        context.keywords(law.rule) if lookup is None else Arguments(),  # type: ignore[arg-type]
        context.fused and lookup is not None and law.schedule == 'synchronous',
    )


def compile_drive(law: Drive, context: LatticeContext) -> CompiledDrive:
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
    return CompiledDrive(
        context.owner,
        context.state(law.target, 'A drive'),
        law.function,
        context.keywords(law.function),
    )


def _compile_part(law: object, context: LatticeContext) -> CompiledRule | CompiledDrive:
    """Return a law of a relaxation compiled."""
    if isinstance(law, LocalRule):
        return compile_rule(law, context)
    if isinstance(law, Drive):
        return compile_drive(law, context)
    error_msg = f'A relaxation of {context.owner} holds {law!r}, which is not a local rule or a drive.'
    raise ValidationError(error_msg)


def compile_relaxation(law: Relaxation, context: LatticeContext) -> CompiledRelaxation:
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
    return CompiledRelaxation(
        law.label or context.owner,
        tuple(_compile_part(item, context) for item in law.drive),
        tuple(_compile_part(item, context) for item in law.relax),
        tuple(context.state(item, 'A relaxation') for item in law.targets),
        law.stable,
        int(law.max_iterations),
        law.activity,
    )


def _neighbor_calls(expression: Expression) -> list[Expression]:
    """Return the `neighbor` calls inside an expression."""
    found, stack = [], [SR(expression)]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if getattr(item.operator(), 'name', lambda: '')() == 'neighbor':
            found.append(item)
            continue
        stack.extend(item.operands())
    return found


def compile_monte_carlo(energy: Energy, move: Move, sampler: Sampler, context: LatticeContext) -> CompiledMonteCarlo:
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
    target = context.state(move.target, 'A move')
    if sampler.kind == 'heat-bath' and move.kind == 'rotate':
        error_msg = f'{owner} use a heat-bath sampler with a rotation, which has no finite set of values to draw from.'
        raise ValidationError(error_msg)
    space = target.space
    if sampler.schedule == 'checkerboard':
        _check_checkerboard(space, energy.neighborhood, owner)
    expression = SR(energy.expression)
    neighbours = [symbol(f'_neighbor_{index}') for index in range(len(energy.neighborhood))]
    replacements = {}
    for call in _neighbor_calls(expression):
        state, index = call.operands()
        if state != move.target:
            error_msg = f'{owner} read the neighbours of {state}, but only the moved state {move.target} has them.'
            raise ValidationError(error_msg)
        replacements[call] = neighbours[int(index)]
    proposal = symbol('_proposal')
    local = expression.subs(replacements)
    change = local.subs({move.target: proposal}) - local
    known = {move.target, proposal, *neighbours}
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
    return CompiledMonteCarlo(
        context.owner,
        target,
        space.index_table(energy.neighborhood),
        numpy_function(time, [move.target, *neighbours, proposal, *extra], [change]),
        arguments,
        numpy_function(time, extra, [SR(sampler.temperature)]),
        move,
        sampler,
    )


def lattice_shape(name: str, steps: Sequence[LatticeStep]) -> tuple[int, ...] | None:
    """Return the lattice shape of a state that lattice steps read, or `None`.

    Args:
        name:
            The dotted name of the state.

        steps:
            The lattice steps.

    Returns:
        shape:
            The lattice shape.
    """
    for step in steps:
        states = (step.target,) if isinstance(step, CompiledMonteCarlo) else step.targets
        for state in states:
            if state.name == name:
                return state.space.dimensions
    return None
