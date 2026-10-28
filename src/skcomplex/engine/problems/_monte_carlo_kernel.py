"""Monte Carlo sampling of a lattice state compiled from its local energy, move, and sampler."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ...errors import SolverError
from ...specs import MoveSpec, SamplerSpec
from ...types import Array, Indices
from ._arguments import Arguments
from ._base import BaseKernel
from ._lattice_state import LatticeState
from ._state import State


class MonteCarloKernel(BaseKernel):
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

    def __init__(  # noqa: PLR0917
        self,
        owner: str,
        target: LatticeState,
        table: Indices,
        delta: Callable[..., Any],
        arguments: Arguments,
        temperature: Callable[..., Any],
        move: MoveSpec,
        sampler: SamplerSpec,
        classes: tuple[Indices, ...] = (),
        masked: bool = False,
    ) -> None:
        self.owner = owner
        self.target = target
        self.table = table
        self.delta = delta
        self.arguments = arguments
        self.temperature = temperature
        self.move = move
        self.sampler = sampler
        self.classes = classes
        self.masked = masked

    def _energy_change(
        self,
        x: Array,
        values: Array,
        sites: Indices,
        proposal: Array,
        keywords: dict[str, Any],
    ) -> Array:
        """Return the energy change of moving some sites to proposed values."""
        rows = self.table[sites]
        neighbours = np.append(x[self.target.sites], self.target.space.fill)[rows]
        masks = (rows != self.target.size).astype(float).T if self.masked else ()
        change = self.delta(0.0, values, *neighbours.T, proposal, *masks, *keywords.values())[0]
        return np.broadcast_to(np.asarray(change, dtype=float), values.shape)

    def _propose(self, values: Array, rng: np.random.Generator) -> Array:
        """Return proposed values for some sites."""
        if self.move.kind == 'flip':
            return -values
        if self.move.kind == 'random-value':
            return np.asarray(self.move.values)[rng.integers(len(self.move.values), size=values.size)]
        return values + rng.uniform(-self.move.width, self.move.width, size=values.size)

    def _update(self, x: Array, p: Array, sites: Indices, rng: np.random.Generator) -> None:
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

    def _run(self, inputs: State) -> None:
        """Update the state vector of the state in place by one sweep of the lattice."""
        x, p, rng = inputs.x, inputs.p, cast('np.random.Generator', inputs.rng)
        size = self.target.size
        if self.sampler.schedule in {'checkerboard', 'colored'}:
            for sites in self.classes:
                self._update(x, p, sites, rng)
            return
        for site in rng.permutation(size):
            self._update(x, p, np.asarray([site]), rng)
