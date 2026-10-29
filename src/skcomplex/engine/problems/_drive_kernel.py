"""Drive compiled for a lattice state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any

import numpy as np

from ...core.specs import LatticeStateSpec
from ...specs import ArgumentsSpec, RunStateSpec
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
    """

    def __init__(
        self,
        owner: str,
        target: LatticeStateSpec,
        function: Callable[..., Any],
        arguments: ArgumentsSpec = _NO_ARGUMENTS,
    ) -> None:
        self.owner = owner
        self.target = target
        self.function = function
        self.arguments = arguments

    @property
    def targets(self) -> tuple[LatticeStateSpec, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def _run(self, inputs: RunStateSpec) -> None:
        """Update the state vector of the state in place by one application of the drive."""
        x, p, rng = inputs.x, inputs.p, inputs.rng
        shape = self.target.shape
        keywords = {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.values(x, p).items()
        }
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, random=rng, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()
