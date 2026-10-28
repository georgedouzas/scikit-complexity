"""Drive compiled for a lattice state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...types import Array
from ._arguments import Arguments
from ._lattice_state import LatticeState


@dataclass(frozen=True)
class CompiledDrive:
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

    owner: str
    target: LatticeState
    function: Callable[..., Any]
    arguments: Arguments = field(default_factory=Arguments)

    @property
    def targets(self) -> tuple[LatticeState, ...]:
        """Lattice states the drive changes."""
        return (self.target,)

    def apply(self, x: Array, p: Array, rng: np.random.Generator) -> None:
        """Update the state vector in place by one application of the drive.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.
        """
        shape = self.target.shape
        keywords = {
            name: np.reshape(value, shape) if np.size(value) == self.target.size and np.ndim(value) else value
            for name, value in self.arguments.values(x, p).items()
        }
        values = np.reshape(x[self.target.sites].copy(), shape)
        result = self.function(values, random=rng, **keywords)
        x[self.target.sites] = np.broadcast_to(np.asarray(result, dtype=float), shape).ravel()
