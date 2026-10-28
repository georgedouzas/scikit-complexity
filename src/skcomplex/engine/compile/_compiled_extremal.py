"""Extremal law compiled for a lattice or a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import SolverError
from ...types import Array, Indices
from ._arguments import Arguments
from ._extremal_record import ExtremalRecord
from ._lattice_state import LatticeState


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

    def _apply_once(self, x: Array, p: Array, rng: np.random.Generator) -> tuple[int, float]:
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

    def apply(self, x: Array, p: Array, rng: np.random.Generator, time: float) -> list[ExtremalRecord]:
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
