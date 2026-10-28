"""Reductions of a problem, evaluated in dependency order into the extended parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.types import Vector
from ._neighborhood_kernel import NeighborhoodKernel
from ._reduction_kernel import ReductionKernel
from ._state import State


@dataclass
class ReductionSchedule:
    """Reductions of a problem, evaluated in dependency order into the extended parameter vector.

    Args:
        reductions:
            The compiled reductions over groups, every one after the reductions it reads.

        parameters:
            The size of the parameter vector, where the reductions start.

        neighbourhoods:
            The compiled reductions over the neighbours of every member.

        sequence:
            Every reduction of both kinds in dependency order, or empty for the reductions over groups in their order.
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

    def extend(self, t: float, x: Vector, p: Vector) -> Vector:
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
            reduction.run(State(t, x, q))
        return q

    def wrap(self, function: Callable[[float, Vector, Vector], Any] | None) -> Callable[..., Any] | None:
        """Return a function of the state and the parameters that reads the reductions at the state it receives.

        Args:
            function:
                A function `f(t, x, q)` of the extended parameter vector, or `None`.

        Returns:
            wrapped:
                The function `f(t, x, p)`, or the function itself without reductions.
        """
        if function is None or not self.steps:
            return function

        def reduced(t: float, x: Vector, p: Vector) -> Any:  # noqa: ANN401
            return function(t, x, self.extend(t, x, p))

        return reduced
