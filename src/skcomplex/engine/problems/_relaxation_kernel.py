"""Relaxation compiled for a lattice, with its drive and relaxation laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.specs import LatticeStateSpec
from ...errors import SolverError
from ...specs import RelaxationRecordSpec, RunStateSpec
from ...types import Array
from ._base import BaseKernel
from ._drive_kernel import DriveKernel
from ._rule_kernel import RuleKernel


class RelaxationKernel(BaseKernel):
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

    def __init__(  # noqa: PLR0917
        self,
        label: str,
        drive: tuple[RuleKernel | DriveKernel, ...],
        relax: tuple[RuleKernel | DriveKernel, ...],
        targets: tuple[LatticeStateSpec, ...],
        stable: Callable[..., bool],
        max_iterations: int,
        activity: Callable[..., Any] | None = None,
    ) -> None:
        self.label = label
        self.drive = drive
        self.relax = relax
        self.targets = targets
        self.stable = stable
        self.max_iterations = max_iterations
        self.activity = activity

    def _arrays(self, x: Array) -> list[Array]:
        """Return the targets' arrays in lattice shape."""
        return [np.reshape(x[state.sites], state.shape) for state in self.targets]

    def _is_stable(self, x: Array) -> bool:
        """Check the stability condition on the current state."""
        return bool(self.stable(*self._arrays(x)))

    def _run(self, inputs: RunStateSpec) -> RelaxationRecordSpec | None:
        """Apply the drive, then relax until stable, and return the record of a relaxation that changed the lattice.

        The state vector of the state is updated in place, and the record holds the time of the state.

        Raises:
            SolverError:
                The lattice is not stable after the largest number of iterations.
        """
        x, time = inputs.x, inputs.t
        for law in self.drive:
            law.run(inputs)
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
                if active is not None and isinstance(law, RuleKernel) and law.schedule == 'synchronous':
                    law._run_near(inputs, active)
                else:
                    law.run(inputs)
            updated = (x[sites] != before).reshape(len(self.targets), -1).any(axis=0) if active is None else active
            size += int(updated.sum())
            changed = updated if changed is None else changed | updated
            duration += 1
        if not duration:
            return None
        area = int(changed.sum()) if changed is not None else 0
        return RelaxationRecordSpec(self.label, time, size, duration, area)
