"""Relaxation compiled for a lattice, with its drive and relaxation laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.errors import SolverError
from ._compiled_drive import CompiledDrive
from ._compiled_rule import CompiledRule
from ._lattice_state import LatticeState
from ._lookup import Vector
from ._relaxation_record import RelaxationRecord


@dataclass(frozen=True)
class CompiledRelaxation:
    """Relaxation compiled for a lattice, with its drive and relaxation laws.

    Args:
        label:
            The label of its events.

        drive:
            The laws applied once per step.

        relax:
            The laws applied until the lattice is stable.

        targets:
            The states the stability condition reads.

        stable:
            The stability condition on the targets' arrays in lattice shape.

        max_iterations:
            The largest number of relaxation iterations in one step.

        activity:
            The function that marks the active sites of an iteration, or `None` to count the sites that change.
    """

    label: str
    drive: tuple[CompiledRule | CompiledDrive, ...]
    relax: tuple[CompiledRule | CompiledDrive, ...]
    targets: tuple[LatticeState, ...]
    stable: Callable[..., bool]
    max_iterations: int
    activity: Callable[..., Any] | None = None

    def _arrays(self, x: Vector) -> list[Vector]:
        """Return the targets' arrays in lattice shape."""
        return [np.reshape(x[state.sites], state.shape) for state in self.targets]

    def _is_stable(self, x: Vector) -> bool:
        """Check the stability condition on the current state."""
        return bool(self.stable(*self._arrays(x)))

    def apply(self, x: Vector, p: Vector, rng: np.random.Generator, time: float) -> RelaxationRecord | None:
        """Apply the drive, then relax until stable, and return the record of a relaxation that changed the lattice.

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
            record:
                The size, duration, and area of the relaxation, or `None` when the lattice was already stable.

        Raises:
            SolverError:
                The lattice is not stable after the largest number of iterations.
        """
        for law in self.drive:
            law.apply(x, p, rng)
        size = duration = 0
        changed: npt.NDArray[np.bool_] | None = None
        sites = np.concatenate([np.arange(state.start, state.start + state.size) for state in self.targets])
        while not self._is_stable(x):
            if duration >= self.max_iterations:
                error_msg = (
                    f'The relaxation {self.label} is not stable after {self.max_iterations} iterations. Check its '
                    'stability condition or raise max_iterations.'
                )
                raise SolverError(error_msg)
            before = x[sites].copy()
            active = (
                np.ravel(np.asarray(self.activity(*self._arrays(x)), dtype=bool)) if self.activity is not None else None
            )
            for law in self.relax:
                if active is not None and isinstance(law, CompiledRule) and law.schedule == 'synchronous':
                    law.apply_near(x, p, rng, active)
                else:
                    law.apply(x, p, rng)
            updated = (x[sites] != before).reshape(len(self.targets), -1).any(axis=0) if active is None else active
            size += int(updated.sum())
            changed = updated if changed is None else changed | updated
            duration += 1
        if not duration:
            return None
        area = int(changed.sum()) if changed is not None else 0
        return RelaxationRecord(self.label, time, size, duration, area)
