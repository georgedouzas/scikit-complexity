"""Monte Carlo sampling of a lattice state compiled from its local objective, move, and sampler."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

import numpy as np
import scipy.sparse

from ...core.specs import LatticeStateSpec
from ...errors import SolverError
from ...specs import ArgumentsSpec, MoveSpec, RunStateSpec, SamplerSpec
from ...types import Array, Indices
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter


class MonteCarloKernel(BaseKernel):
    """Monte Carlo sampling of a lattice state compiled from its local objective, move, and sampler.

    Args:
        owner:
            The path of the component that declares the objective.

        target:
            The lattice state the move changes.

        table:
            The member index of every neighbour of every site.

        delta:
            The objective change `delta(value, *neighbours, proposal, **arguments)` over arrays of sites.

        arguments:
            The parameters and states the objective and the intensity read.

        intensity:
            The intensity of the acceptance as a function of the arguments.

        move:
            The move.

        sampler:
            The acceptance rule and the schedule.

        classes:
            The sites of every class of the colored or checkerboard schedule.

        masked:
            Whether the objective change reads the mask of the real neighbours of a neighbour sum.

        bond:
            For cluster moves, the half change `bond(value, neighbour, **arguments)` of the objective of a site with one
            neighbour when the value of the site flips and then the value of the neighbour flips, or `None`.

        boundary:
            For a local objective over cofaces, the incidence of the cells of the state and their cofaces, or `None`.

        signs:
            For a local objective over cofaces, the sign of every coface of every cell, which `table` then lists.
    """

    values_: None

    def __init__(
        self,
        owner: str,
        target: LatticeStateSpec,
        table: Indices,
        delta: Callable[..., object],
        arguments: ArgumentsSpec,
        *,
        intensity: Callable[..., object],
        move: MoveSpec,
        sampler: SamplerSpec,
        classes: tuple[Indices, ...] = (),
        masked: bool = False,
        bond: Callable[..., object] | None = None,
        boundary: scipy.sparse.csr_array | None = None,
        signs: Array | None = None,
    ) -> None:
        self.owner = owner
        self.target = target
        self.table = table
        self.delta = delta
        self.arguments = arguments
        self.intensity = intensity
        self.move = move
        self.sampler = sampler
        self.classes = classes
        self.masked = masked
        self.bond = bond
        self.boundary = boundary
        self.signs = signs

    def _objective_change(
        self,
        x: Array,
        values: Array,
        sites: Indices,
        proposal: Array,
        keywords: Mapping[str, object],
    ) -> Array:
        """Return the objective change of moving some sites to proposed values."""
        rows = self.table[sites]
        if self.boundary is None or self.signs is None:
            neighbours = np.append(x[self.target.sites], self.target.space.fill)[rows]
            masks = (rows != self.target.size).astype(float).T if self.masked else ()
        else:
            signs = self.signs[sites]
            around = np.append(self.boundary @ x[self.target.sites], 0.0)[rows] - signs * values[:, None]
            neighbours = np.hstack([around, signs])
            masks = (rows != self.boundary.shape[0]).astype(float).T
        change = ExpressionPrinter._values_of(
            self.delta(0.0, values, *neighbours.T, proposal, *masks, *keywords.values()),
        )[0]
        return np.broadcast_to(np.asarray(change, dtype=float), values.shape)

    def _propose(self, values: Array, rng: np.random.Generator) -> Array:
        """Return proposed values for some sites."""
        if self.move.kind == 'flip':
            return -values
        if self.move.kind == 'random-value':
            return np.asarray(self.move.values)[rng.integers(len(self.move.values), size=values.size)]
        if self.move.kind == 'gaussian':
            return values + rng.normal(0.0, self.move.width, size=values.size)
        if self.move.kind == 'function' and self.move.function is not None:
            return np.broadcast_to(np.asarray(self.move.function(values.copy(), rng), dtype=float), values.shape)
        return values + rng.uniform(-self.move.width, self.move.width, size=values.size)

    def _intensity_of(self, keywords: Mapping[str, object]) -> Array:
        """Return the intensity at some sites, infinite where it divides by zero, after checking it is positive."""
        arguments = [np.asarray(value, dtype=float) for value in keywords.values()]
        with np.errstate(divide='ignore'):
            value = ExpressionPrinter._values_of(self.intensity(self.time_, *arguments))[0]
        intensity = np.asarray(value, dtype=float)
        if np.any(intensity <= 0):
            error_msg = f'The intensity of {self.owner} is {intensity}, but it must be positive.'
            raise SolverError(error_msg)
        return intensity

    @staticmethod
    def _scaled(change: Array, intensity: Array) -> Array:
        """Return a change of the objective times the intensity, 0 for no change at an infinite intensity."""
        return np.where(change == 0.0, 0.0, change * np.where(change == 0.0, 1.0, intensity))

    def _grow(self, x: Array, p: Array, values: Array, sites: Indices, rng: np.random.Generator) -> Indices:
        """Return the neighbours that bonds add to a cluster from its newest sites."""
        rows = self.table[sites]
        keywords = {
            name: value[:, None] if isinstance(value, np.ndarray) else value
            for name, value in self.arguments.values(x, p, sites).items()
        }
        intensity = self._intensity_of(keywords)
        neighbours = np.append(values, self.target.space.fill)[rows]
        if self.bond is None:
            error_msg = f'The Monte Carlo laws of {self.owner} make cluster moves without the objective of a bond.'
            raise SolverError(error_msg)
        bond = ExpressionPrinter._values_of(self.bond(0.0, values[sites][:, None], neighbours, *keywords.values()))[0]
        bonds = np.broadcast_to(np.asarray(bond, dtype=float), rows.shape)
        with np.errstate(over='ignore'):
            probability = 1.0 - np.exp(np.minimum(self._scaled(bonds, intensity), 0.0))
        added = (rows != self.target.size) & (rng.random(rows.shape) < probability)
        return np.unique(rows[added])

    def _flip_cluster(self, x: Array, p: Array, rng: np.random.Generator) -> int:
        """Flip the values of a cluster grown by Wolff bonds from a random site, and return its size."""
        start, size = self.target.start, self.target.size
        values = x[start : start + size].copy()
        inside = np.zeros(size, dtype=bool)
        newest = rng.integers(size, size=1)
        inside[newest] = True
        while newest.size:
            found = self._grow(x, p, values, newest, rng)
            newest = found[~inside[found]]
            inside[newest] = True
        x[start + np.flatnonzero(inside)] = -values[inside]
        return int(inside.sum())

    def _update(self, x: Array, p: Array, sites: Indices, rng: np.random.Generator) -> None:
        """Propose and accept moves at sites that do not neighbour each other."""
        keywords = self.arguments.values(x, p, sites)
        intensity = self._intensity_of(keywords)
        values = x[self.target.start + sites]
        if self.sampler.kind == 'heat-bath':
            candidates = (
                np.unique(np.concatenate([values, -values]))
                if self.move.kind == 'flip'
                else np.asarray(self.move.values)
            )
            objectives = np.stack(
                [
                    self._objective_change(x, values, sites, np.full(values.shape, value), keywords)
                    for value in candidates
                ],
            )
            weights = np.exp(-self._scaled(objectives - objectives.min(axis=0), intensity))
            cumulative = np.cumsum(weights / weights.sum(axis=0), axis=0)
            chosen = (rng.random(values.size) > cumulative).sum(axis=0)
            x[self.target.start + sites] = candidates[np.minimum(chosen, candidates.size - 1)]
            return
        proposal = self._propose(values, rng)
        change = self._objective_change(x, values, sites, proposal, keywords)
        with np.errstate(over='ignore'):
            if self.sampler.kind == 'metropolis':
                probability = np.exp(-self._scaled(np.maximum(change, 0.0), intensity))
            else:
                probability = 1.0 / (1.0 + np.exp(self._scaled(change, intensity)))
        accepted = rng.random(values.size) < probability
        x[self.target.start + sites[accepted]] = proposal[accepted]

    @property
    def targets(self) -> tuple[LatticeStateSpec, ...]:
        """Lattice state the move changes, as the one target of the step."""
        return (self.target,)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Update the state vector of the state in place by one sweep of the lattice."""
        self.values_ = None
        state = self._state_of(inputs)
        x, p, rng = state.x, state.p, self._generator_of(state)
        self.time_ = state.t
        size = self.target.size
        if self.move.kind == 'cluster':
            flipped = 0
            while flipped < size:
                flipped += self._flip_cluster(x, p, rng)
            return
        if self.sampler.schedule in {'checkerboard', 'colored'}:
            for sites in self.classes:
                self._update(x, p, sites, rng)
            return
        for site in rng.permutation(size):
            self._update(x, p, np.asarray([site]), rng)
