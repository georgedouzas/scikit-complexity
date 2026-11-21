"""Define the reduction schedule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...base import BaseSpec
from ...types import Array
from ._neighborhood_kernel import NeighborhoodKernel
from ._reduction_kernel import ReductionKernel
from ._run_state_spec import RunStateSpec


@dataclass(frozen=True)
class ReductionScheduleSpec(BaseSpec):
    """A spec of the reductions of a problem, in the order they must run.

    A reduction may read another, such as a variance that reads a mean, so the spec lists every reduction over a group
    or over neighbours after those it reads, and says where their values start after the parameters. The engine runs
    them in that order before the laws at every evaluation. It orders the mean field of oscillators, the totals of a
    market, and the counts of an epidemic.

    Read more in the user guide, under Extend the library.

    Args:
        reductions:
            The compiled reductions over groups, every one after the reductions it reads.

        parameters:
            The size of the parameter vector, where the reductions start.

        neighbourhoods:
            The compiled reductions over the neighbours of every member.

        sequence:
            Every reduction of both kinds in dependency order, or empty for the reductions over groups in their order.

    See Also:
        ReductionKernel: One reduction over a group.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import KuramotoSystem
        >>> schedule = KuramotoSystem().prepare().simulate(until=0).problem.jacobian.reductions
        >>> [kernel.kind for kernel in schedule.reductions]
        ['count', 'sum', 'sum']
    """

    reductions: tuple[ReductionKernel, ...] = ()
    parameters: int = 0
    neighbourhoods: tuple[NeighborhoodKernel, ...] = ()
    sequence: tuple[ReductionKernel | NeighborhoodKernel, ...] = ()

    @property
    def steps(self) -> tuple[ReductionKernel | NeighborhoodKernel, ...]:
        """Every reduction in the order it is evaluated."""
        return self.sequence or (*self.reductions, *self.neighbourhoods)

    @property
    def extent(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.steps), default=self.parameters)

    def extend(self, t: float, x: Array, p: Array) -> Array:
        """Return the parameter vector followed by the value of every reduction at a state.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            extended:
                The extended parameter vector.
        """
        q = np.zeros(self.extent)
        given = min(p.size, self.parameters)
        q[:given] = p[:given]
        for reduction in self.steps:
            reduction.run(RunStateSpec(t, x, q))
        return q

    def wrap[R](
        self,
        function: Callable[[float, Array, Array], R] | None,
    ) -> Callable[[float, Array, Array], R] | None:
        """Return a function of the state and the parameters.

        The function reads the reductions at the state it receives.

        Args:
            function:
                A function `f(t, x, q)` of the extended parameter vector, or `None`.

        Returns:
            wrapped:
                The function `f(t, x, p)`, or the function itself without reductions.
        """
        if function is None or not self.steps:
            return function

        def reduced(t: float, x: Array, p: Array) -> R:
            return function(t, x, self.extend(t, x, p))

        return reduced
