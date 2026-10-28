"""Extremal law compiled for a lattice or a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from ...errors import SolverError
from ...specs import ArgumentsSpec, ExtremalRecordSpec, LatticeStateSpec, RunStateSpec
from ...types import Array, Indices
from ._base import BaseKernel


class ExtremalKernel(BaseKernel):
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

    def __init__(  # noqa: PLR0917
        self,
        label: str,
        owner: str,
        targets: tuple[LatticeStateSpec, ...],
        key: LatticeStateSpec,
        members: Indices,
        counts: Indices,
        rule: Callable[..., Any],
        arguments: ArgumentsSpec,
        maximum: bool,
        repeat: int,
        loop: Callable[..., None] | None = None,
    ) -> None:
        self.label = label
        self.owner = owner
        self.targets = targets
        self.key = key
        self.members = members
        self.counts = counts
        self.rule = rule
        self.arguments = arguments
        self.maximum = maximum
        self.repeat = repeat
        self.loop = loop

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

    def _run(self, inputs: RunStateSpec) -> list[ExtremalRecordSpec]:
        """Apply the law `repeat` times to the state vector of the state in place, and return the records.

        Every record holds the time of the state, the site, and the extremal value of one application.
        """
        x, p, rng, time = inputs.x, inputs.p, cast('np.random.Generator', inputs.rng), inputs.t
        if self.loop is None:
            found = [self._apply_once(x, p, rng) for _ in range(self.repeat)]
            return [ExtremalRecordSpec(self.label, time, site, value) for site, value in found]
        sites, values = np.zeros(self.repeat, dtype=np.intp), np.zeros(self.repeat)
        self.loop(x, np.ascontiguousarray(p, dtype=float), rng, self.members, self.counts, self.repeat, sites, values)
        return [
            ExtremalRecordSpec(self.label, time, int(site), float(value))
            for site, value in zip(sites.tolist(), values.tolist(), strict=True)
        ]
