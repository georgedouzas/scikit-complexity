"""Drive compiled for a lattice state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np

from ...types import Array
from ._arguments_spec import _ArgumentsSpec
from ._base import BaseKernel
from ._lattice_state_spec import _LatticeStateSpec
from ._run_state_spec import RunStateSpec

NO_ARGUMENTS = _ArgumentsSpec()


class DriveKernel(BaseKernel):
    """A kernel of a Python function that changes a whole state on a lattice at once.

    The function receives the array of the state in the shape of the lattice, with the components of a member on a last
    axis, and may also receive the time, the random generator, and the space, such as a function that drops one grain of
    sand on a random site. It runs between the steps of a cascade or of a local rule. It drops grains on a sandpile,
    lights fires in a forest, and plants seeds on a field.

    Read more in the user guide, under Extend the library.

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

        spaced:
            Whether the function receives the space of the state by the keyword `space`.

    See Also:
        CascadeKernel: Runs a drive, then relaxes the lattice.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SandpileSystem
        >>> kernel = SandpileSystem().prepare().simulate(until=0).problem.lattice[0].drive[0]
        >>> type(kernel).__name__, kernel.drawing
        ('DriveKernel', True)
    """

    values_: None

    def __init__(
        self,
        owner: str,
        target: _LatticeStateSpec,
        function: Callable[..., object],
        arguments: _ArgumentsSpec = NO_ARGUMENTS,
        *,
        timed: bool = False,
        drawing: bool = True,
        score: Callable[..., object] | None = None,
        spaced: bool = False,
    ) -> None:
        self.owner = owner
        self.target = target
        self.function = function
        self.arguments = arguments
        self.timed = timed
        self.drawing = drawing
        self.score = score
        self.spaced = spaced

    @property
    def targets(self) -> tuple[_LatticeStateSpec, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def _get_keywords(self, x: Array, p: Array) -> dict[str, object]:
        """Return the parameters the functions receive by keyword.

        Those of one value per site come in lattice shape.
        """
        shape = self.target.shape
        return {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.get_values(x, p).items()
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
        return self.score(np.reshape(x[self.target.sites].copy(), self.target.shape), **self._get_keywords(x, p))

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Update the state vector of the state in place by one application of the drive."""
        self.values_ = None
        state = self._get_state_of(inputs)
        x, p, rng = state.x, state.p, state.rng
        shape = self.target.shape if self.target.components == 1 else (*self.target.shape, self.target.components)
        keywords = self._get_keywords(x, p)
        if self.timed:
            keywords['time'] = state.t
        if self.drawing:
            keywords['random'] = rng
        if self.spaced:
            keywords['space'] = self.target.space
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()
