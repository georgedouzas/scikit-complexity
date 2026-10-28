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
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ...core.errors import SolverError, ValidationError
from ...core.spaces import BaseSpace
from ...core.specs import MoveSpec, SamplerSpec

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]

_LOOKUP_KINDS = {'table': 0, 'totalistic': 1, 'outer': 2}
_COLORING_STRATEGY = 'largest_first'
type LatticeStep = CompiledRule | CompiledDrive | CompiledRelaxation | CompiledMonteCarlo | CompiledExtremal


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


@dataclass(frozen=True)
class LatticeState:
    """State of a lattice group in the state vector.

    Args:
        name:
            The dotted name of the state.

        start:
            The index of its first component.

        space:
            The lattice of the group, or the network whose nodes are its sites.
    """

    name: str
    start: int
    space: BaseSpace

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of the sites of the group."""
        return self.space.shape_

    @functools.cached_property
    def size(self) -> int:
        """Number of sites of the group."""
        return int(np.prod(self.shape))

    @functools.cached_property
    def sites(self) -> slice:
        """Slice of the group's components in the state vector."""
        return slice(self.start, self.start + self.size)


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

        degrees:
            The number of real neighbours of every site, for a function rule that reads them, or `None`.

        classes:
            The sites of every class of the colored or checkerboard schedule.
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
    degrees: Vector | None = None
    classes: tuple[Indices, ...] = ()

    @property
    def space(self) -> BaseSpace:
        """Lattice or network of the rule."""
        return self.targets[0].space

    @functools.cached_property
    def reading(self) -> scipy.sparse.csr_matrix:
        """Sparse matrix whose entry `(i, j)` is 1 when site `i` reads site `j`."""
        size = self.table.shape[0]
        rows = np.repeat(np.arange(size), self.table.shape[1])
        columns = self.table.ravel()
        real = columns < size
        ones = np.ones(int(real.sum()))
        return scipy.sparse.csr_matrix((ones, (rows[real], columns[real])), shape=(size, size))

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
                    _LOOKUP_KINDS[lookup.kind],
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
        if self.degrees is not None:
            keywords['degree'] = self.degrees if sites is None else self.degrees[sites]
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

    def apply_near(self, x: Vector, p: Vector, rng: np.random.Generator, active: npt.NDArray[np.bool_]) -> None:
        """Update in place, synchronously, the active sites and the sites that read one of them, and no other site.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.

            active:
                Whether every site is active.
        """
        readers = active | (self.reading @ active.astype(np.float64) > 0)
        sites = np.flatnonzero(readers)
        if sites.size:
            self._assign(x, self._next(x, p, sites, rng), sites)

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
        if self.schedule in {'checkerboard', 'colored'}:
            for sites in self.classes:
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

        others:
            The entry name of every symbol of a neighbour, the namespace `s.other` of the laws.
    """

    owner: str
    states: Mapping[Expression, LatticeState]
    arguments: Mapping[Expression, tuple[str, bool, int, int]]
    fused: bool = False
    others: Mapping[Expression, str] = field(default_factory=dict)

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
        names = self._accepted(function, list(by_name))
        return Arguments(
            names,
            tuple(by_name[name][0] for name in names),
            tuple(by_name[name][1] for name in names),
            tuple(by_name[name][2] for name in names),
        )

    @staticmethod
    def _accepted(function: Callable[..., Any], names: Sequence[str]) -> tuple[str, ...]:
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
                if active is not None and isinstance(law, CompiledRule) and law.schedule == 'synchronous':
                    law.apply_near(x, p, rng, active)
                else:
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

        classes:
            The sites of every class of the colored or checkerboard schedule.

        masked:
            Whether the energy change reads the mask of the real neighbours of a neighbour sum.
    """

    owner: str
    target: LatticeState
    table: Indices
    delta: Callable[..., Any]
    arguments: Arguments
    temperature: Callable[..., Any]
    move: MoveSpec
    sampler: SamplerSpec
    classes: tuple[Indices, ...] = ()
    masked: bool = False

    def _energy_change(
        self,
        x: Vector,
        values: Vector,
        sites: Indices,
        proposal: Vector,
        keywords: dict[str, Any],
    ) -> Vector:
        """Return the energy change of moving some sites to proposed values."""
        rows = self.table[sites]
        neighbours = np.append(x[self.target.sites], self.target.space.fill)[rows]
        masks = (rows != self.target.size).astype(float).T if self.masked else ()
        change = self.delta(0.0, values, *neighbours.T, proposal, *masks, *keywords.values())[0]
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

    @property
    def targets(self) -> tuple[LatticeState, ...]:
        """Lattice state the move changes, as the one target of the step."""
        return (self.target,)

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
        if self.sampler.schedule in {'checkerboard', 'colored'}:
            for sites in self.classes:
                self._update(x, p, sites, rng)
            return
        for site in rng.permutation(size):
            self._update(x, p, np.asarray([site]), rng)


@dataclass(frozen=True, slots=True)
class ExtremalRecord:
    """Application of an extremal law, such as the replacement of the least fit species.

    Args:
        label:
            The label of the extremal law.

        time:
            The time of the step.

        site:
            The member index of the extremal site, in row-major order on a lattice or node order on a network.

        value:
            The extremal value of the key state before the update.
    """

    label: str
    time: float
    site: int
    value: float


@dataclass(frozen=True)
class CompiledExtremal:
    """Extremal law compiled for a lattice or a network.

    Args:
        label:
            The label of its events.

        owner:
            The path of the component that declares the law.

        targets:
            The lattice states the rule updates.

        key:
            The lattice state whose extremum is searched.

        members:
            Every site followed by its real neighbours without duplicates, padded with -1.

        counts:
            The number of sites the rule updates around every site.

        rule:
            The rule.

        arguments:
            The parameters the rule receives by keyword.

        maximum:
            Whether the law searches the largest value rather than the smallest.

        repeat:
            The number of applications per step.

        loop:
            The compiled loop of the applications on the Numba backend, or `None` for the NumPy loop.
    """

    label: str
    owner: str
    targets: tuple[LatticeState, ...]
    key: LatticeState
    members: Indices
    counts: Indices
    rule: Callable[..., Any]
    arguments: Arguments
    maximum: bool
    repeat: int
    loop: Callable[..., None] | None = None

    def _apply_once(self, x: Vector, p: Vector, rng: np.random.Generator) -> tuple[int, float]:
        """Update the extremal site and its neighbours once, and return the site and its value."""
        keys = x[self.key.sites]
        best = keys.max() if self.maximum else keys.min()
        ties = np.flatnonzero(keys == best)
        site = int(ties[rng.integers(0, ties.size)] if ties.size > 1 else ties[0])
        sites = self.members[site, : self.counts[site]]
        random = rng.random(sites.size)
        values = [x[state.start + sites] for state in self.targets]
        result = self.rule(*values, random=random, **self.arguments.values(x, p, sites))
        results = list(result) if len(self.targets) > 1 else [result]
        if len(results) != len(self.targets):
            error_msg = (
                f'The extremal law of {self.owner} returned {len(results)} arrays for {len(self.targets)} targets.'
            )
            raise SolverError(error_msg)
        for state, value in zip(self.targets, results, strict=True):
            x[state.start + sites] = value
        return site, float(best)

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator, time: float) -> list[ExtremalRecord]:
        """Apply the law `repeat` times and return the record of every application.

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
            records:
                The site and the extremal value of every application.
        """
        if self.loop is None:
            found = [self._apply_once(x, p, rng) for _ in range(self.repeat)]
            return [ExtremalRecord(self.label, time, site, value) for site, value in found]
        sites, values = np.zeros(self.repeat, dtype=np.intp), np.zeros(self.repeat)
        self.loop(x, np.ascontiguousarray(p, dtype=float), rng, self.members, self.counts, self.repeat, sites, values)
        return [
            ExtremalRecord(self.label, time, int(site), float(value))
            for site, value in zip(sites.tolist(), values.tolist(), strict=True)
        ]


