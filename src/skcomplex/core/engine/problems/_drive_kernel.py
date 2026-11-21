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
    """A kernel of a drive compiled for a lattice state.

    The components of every member lie on a last axis when the state has several. The function and the score receive
    the keywords of every rule of an `UpdateLaw`: the parameters by name, `time`, `space`, `degree`, which is `None`,
    and `random`, the generator for a stochastic drive and `None` otherwise.

    Args:
        owner:
            The path of the component that declares the drive.

        target:
            The lattice state the drive changes.

        function:
            The function of the state's array in lattice shape.

        arguments:
            The parameters the function receives by keyword.

        stochastic:
            Whether the function draws numbers, so that it receives the random generator as `random`.

        score:
            The function of the array of the state and the keywords that gives the score of every row, or `None`.
    """

    values_: None

    def __init__(
        self,
        owner: str,
        target: _LatticeStateSpec,
        function: Callable[..., object],
        arguments: _ArgumentsSpec = NO_ARGUMENTS,
        *,
        stochastic: bool = False,
        score: Callable[..., object] | None = None,
    ) -> None:
        self.owner = owner
        self.target = target
        self.function = function
        self.arguments = arguments
        self.stochastic = stochastic
        self.score = score

    @property
    def targets(self) -> tuple[_LatticeStateSpec, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def _get_keywords(self, t: float, x: Array, p: Array, random: np.random.Generator | None) -> dict[str, object]:
        """Return the keywords the functions receive.

        Parameters of one value per site come in lattice shape.
        """
        shape = self.target.shape
        parameters = {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.get_values(x, p).items()
        }
        return {**parameters, 'time': t, 'space': self.target.space, 'degree': None, 'random': random}

    def read_score(self, t: float, x: Array, p: Array) -> object:
        """Return the score of every row of the state.

        Args:
            t:
                The time.

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
        values = np.reshape(x[self.target.sites].copy(), self.target.shape)
        return self.score(values, **self._get_keywords(t, x, p, None))

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Update the state vector of the state in place by one application of the drive."""
        self.values_ = None
        state = self._get_state_of(inputs)
        x, p = state.x, state.p
        shape = self.target.shape if self.target.components == 1 else (*self.target.shape, self.target.components)
        keywords = self._get_keywords(state.t, x, p, state.rng if self.stochastic else None)
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()
