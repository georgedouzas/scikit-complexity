"""Drive compiled for a lattice state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np

from ...core.specs import LatticeStateSpec
from ...specs import ArgumentsSpec, RunStateSpec
from ...types import Array
from ._base import BaseKernel

_NO_ARGUMENTS = ArgumentsSpec()


class DriveKernel(BaseKernel):
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

        timed:
            Whether the function receives the time of the step by the keyword `time`.

        drawing:
            Whether the function receives the random generator by the keyword `random`.

        score:
            The function of the array of the state and the parameters that gives the score of every row, or `None`.
    """

    values_: None

    def __init__(
        self,
        owner: str,
        target: LatticeStateSpec,
        function: Callable[..., object],
        arguments: ArgumentsSpec = _NO_ARGUMENTS,
        *,
        timed: bool = False,
        drawing: bool = True,
        score: Callable[..., object] | None = None,
    ) -> None:
        self.owner = owner
        self.target = target
        self.function = function
        self.arguments = arguments
        self.timed = timed
        self.drawing = drawing
        self.score = score

    @property
    def targets(self) -> tuple[LatticeStateSpec, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def _keywords(self, x: Array, p: Array) -> dict[str, object]:
        """Return the parameters the functions receive by keyword, those of one value per site in lattice shape."""
        shape = self.target.shape
        return {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.values(x, p).items()
        }

    def read_score(self, x: Array, p: Array) -> object:
        """Return the score of every row of the state.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            scores:
                The value of the score function, or `None` for a drive without one.
        """
        if self.score is None:
            return None
        return self.score(np.reshape(x[self.target.sites].copy(), self.target.shape), **self._keywords(x, p))

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Update the state vector of the state in place by one application of the drive."""
        self.values_ = None
        state = self._state_of(inputs)
        x, p, rng = state.x, state.p, state.rng
        shape = self.target.shape
        keywords = self._keywords(x, p)
        if self.timed:
            keywords['time'] = state.t
        if self.drawing:
            keywords['random'] = rng
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()
