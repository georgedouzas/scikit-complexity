"""Local rule compiled for a lattice, with its neighbour table and its evaluation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse

from ...core.spaces import BaseSpace
from ...errors import SolverError
from ...specs import ArgumentsSpec, LatticeStateSpec, LookupSpec, RunStateSpec
from ...types import Array, Indices
from ._base import BaseKernel

_LOOKUP_KINDS = {'table': 0, 'totalistic': 1, 'outer': 2}


@numba.njit(cache=True)
def _fused_lookup(  # noqa: PLR0917
    values: Array,
    centers: Array,
    fill: float,
    table: Indices,
    kind: int,
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


_NO_ARGUMENTS = ArgumentsSpec()


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
    """

    def __init__(  # noqa: PLR0917
        self,
        owner: str,
        targets: tuple[LatticeStateSpec, ...],
        reads: tuple[LatticeStateSpec, ...],
        table: Indices,
        rule: Callable[..., Any] | None,
        lookup: LookupSpec | None,
        schedule: str,
        stochastic: bool,
        arguments: ArgumentsSpec = _NO_ARGUMENTS,
        fused: bool = False,
        degrees: Array | None = None,
        classes: tuple[Indices, ...] = (),
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

    def _next(self, x: Array, p: Array, sites: Indices | None, rng: np.random.Generator) -> list[Array]:
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

    def _assign(self, x: Array, values: list[Array], sites: Indices | None) -> None:
        """Write the next values of the targets at some sites."""
        for state, value in zip(self.targets, values, strict=True):
            if sites is None:
                x[state.sites] = value
            else:
                x[state.start + sites] = value

    def _run_near(self, inputs: RunStateSpec, active: npt.NDArray[np.bool_]) -> None:
        """Update the state in place, synchronously, at the active sites and the sites that read one of them."""
        x, p, rng = inputs.x, inputs.p, cast('np.random.Generator', inputs.rng)
        readers = active | (self.reading @ active.astype(np.float64) > 0)
        sites = np.flatnonzero(readers)
        if sites.size:
            self._assign(x, self._next(x, p, sites, rng), sites)

    def _run(self, inputs: RunStateSpec) -> None:
        """Update the state vector of the state in place by one application of the rule on its schedule."""
        x, p, rng = inputs.x, inputs.p, cast('np.random.Generator', inputs.rng)
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
