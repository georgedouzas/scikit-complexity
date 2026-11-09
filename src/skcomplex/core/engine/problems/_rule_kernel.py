"""Local rule compiled for a lattice, with its neighbour table and its evaluation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Iterable

import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse

from ...components.spaces import BaseSpace
from ...components.specs import LatticeStateSpec
from ...errors import SolverError, ValidationError
from ...specs import ArgumentsSpec, LookupSpec, RunStateSpec
from ...types import Array, Indices
from ._base import BaseKernel

LOOKUP_KINDS = {'table': 0, 'totalistic': 1, 'outer': 2}
NO_ARGUMENTS = ArgumentsSpec()


@numba.njit(cache=True)
def _fused_lookup(
    values: Array,
    centers: Array,
    fill: float,
    table: Indices,
    kind: int,
    *,
    base: int,
    lookup: Array,
    columns: int,
) -> Array:
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


class RuleKernel(BaseKernel):
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

        translations:
            The site of `g h s` for every site `g`, group element `h`, and word `s` of a memory set translated by the
            value of the first read state, or `None` for the fixed table. With it, every step reads the table at the
            values of that state, and a value that names no site reads the fill value.
    """

    values_: None

    def __init__(
        self,
        owner: str,
        targets: tuple[LatticeStateSpec, ...],
        reads: tuple[LatticeStateSpec, ...],
        table: Indices,
        rule: Callable[..., object] | None,
        *,
        lookup: LookupSpec | None,
        schedule: str,
        stochastic: bool,
        arguments: ArgumentsSpec = NO_ARGUMENTS,
        fused: bool = False,
        degrees: Array | None = None,
        classes: tuple[Indices, ...] = (),
        translations: Indices | None = None,
    ) -> None:
        self.owner = owner
        self.targets = targets
        self.reads = reads
        self.table = table
        self.rule = rule
        self.lookup = lookup
        self.schedule = schedule
        self.stochastic = stochastic
        self.arguments = arguments
        self.fused = fused
        self.degrees = degrees
        self.classes = classes
        self.translations = translations

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

    def _get_table_of(self, x: Array, sites: Indices | None) -> Indices:
        """Return the neighbour table of some sites.

        When translated, it is read at the values of the first read state.
        """
        if self.translations is None:
            return self.table if sites is None else self.table[sites]
        size = self.translations.shape[0]
        rows = np.arange(size) if sites is None else sites
        values = np.rint(x[self.reads[0].sites][rows])
        named = (values >= 0) & (values < size)
        table = self.translations[rows, np.where(named, values, 0).astype(np.intp)]
        return np.where(named[:, None], table, size)

    def _gather(self, x: Array, state: LatticeStateSpec, table: Indices) -> list[Array]:
        """Return the values of a state at every neighbour.

        The components of every member lie on a last axis.
        """
        values = x[state.sites].reshape(-1, state.components)
        padded = np.vstack([values, np.full((1, state.components), self.space.fill)])[table]
        return [
            padded[:, column, 0] if state.components == 1 else padded[:, column] for column in range(table.shape[1])
        ]

    def _get_next(self, x: Array, p: Array, sites: Indices | None, rng: np.random.Generator) -> list[Array]:
        """Return the next values of the targets at some sites.

        They are read from the current state.
        """
        space = self.space
        table = self._get_table_of(x, sites)
        if self.lookup is not None:
            values = x[self.reads[0].sites]
            if self.fused and sites is None and self.translations is None:
                centers = x[self.targets[0].sites]
                lookup = self.lookup
                following = _fused_lookup(
                    np.ascontiguousarray(values),
                    np.ascontiguousarray(centers),
                    float(space.fill),
                    table,
                    LOOKUP_KINDS[lookup.kind],
                    base=lookup.base,
                    lookup=lookup.values,
                    columns=lookup.columns,
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
        columns = [column for state in self.reads for column in self._gather(x, state, table)]
        keywords = self.arguments.get_values(x, p, sites)
        if self.degrees is not None:
            keywords['degree'] = self.degrees if sites is None else self.degrees[sites]
        if self.stochastic:
            keywords['random'] = rng.random(table.shape[0])
        if self.rule is None:
            error_msg = f'The local rule of {self.owner} has neither a function nor a lookup.'
            raise ValidationError(error_msg)
        result = self.rule(*columns, **keywords)
        several = len(self.targets) > 1
        results = list(result) if several and isinstance(result, Iterable) else [result]
        if len(results) != len(self.targets):
            error_msg = (
                f'The local rule of {self.owner} returned {len(results)} arrays for {len(self.targets)} targets.'
            )
            raise SolverError(error_msg)
        return [
            np.broadcast_to(
                np.asarray(item, dtype=float),
                (table.shape[0],) if state.components == 1 else (table.shape[0], state.components),
            )
            for state, item in zip(self.targets, results, strict=True)
        ]

    def _assign(self, x: Array, values: list[Array], sites: Indices | None) -> None:
        """Write the next values of the targets at some sites.

        The components of every member are written together.
        """
        for state, value in zip(self.targets, values, strict=True):
            if sites is None:
                x[state.sites] = np.ravel(value)
            else:
                components = state.components
                indices = state.start + components * sites[:, None] + np.arange(components)
                x[indices.ravel()] = np.ravel(value)

    def _run_near(self, inputs: RunStateSpec, active: npt.NDArray[np.bool_]) -> None:
        """Update the state in place, synchronously, near the active sites.

        The update covers the active sites and the sites that read one of them.
        """
        x, p, rng = inputs.x, inputs.p, self._get_generator_of(inputs)
        readers = active | (self.reading @ active.astype(np.float64) > 0)
        sites = np.flatnonzero(readers)
        if sites.size:
            self._assign(x, self._get_next(x, p, sites, rng), sites)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Update the state vector of the state in place by one application of the rule.

        The rule applies on its schedule.
        """
        self.values_ = None
        state = self._get_state_of(inputs)
        x, p, rng = state.x, state.p, self._get_generator_of(state)
        size = self.targets[0].size
        if self.schedule == 'synchronous':
            self._assign(x, self._get_next(x, p, None, rng), None)
            return
        if self.schedule in {'checkerboard', 'colored'}:
            for sites in self.classes:
                self._assign(x, self._get_next(x, p, sites, rng), sites)
            return
        order = rng.permutation(size) if self.schedule == 'random-sequential' else rng.integers(size, size=size)
        for site in order:
            sites = np.asarray([site])
            self._assign(x, self._get_next(x, p, sites, rng), sites)
