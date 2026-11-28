"""Cascade compiled for a lattice, with its drive and the laws that relax it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...errors import SolverError
from ...types import Array
from ._base import BaseKernel
from ._base_record_spec import BaseRecordSpec
from ._drive_kernel import DriveKernel
from ._rule_kernel import RuleKernel, _LatticeState
from ._run_state_spec import RunStateSpec


@dataclass(frozen=True)
class _CascadeRecordSpec(BaseRecordSpec):
    """A spec of the record of a cascade of a lattice within one step.

    Args:
        label:
            The label of the cascade.

        time:
            The time of the step.

        size:
            The number of site updates.

        duration:
            The number of iterations of the cascade.

        area:
            The number of distinct sites that changed.
    """

    label: str
    time: float
    size: int
    duration: int
    area: int


class CascadeKernel(BaseKernel):
    """A kernel of a cascade on a lattice: a drive, then relaxing until stable.

    At every step it applies the drive laws once, such as dropping one grain of sand on a random site, then applies the
    relax laws again and again until the stability condition holds everywhere, such as no site holding four grains or
    more. It records the size and length of the cascade as an event. It runs the avalanches of a sandpile, the fires of
    a forest, and the failures of a power grid.

    Read more in the user guide, under Extend the library.

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
            The largest number of iterations in one step.

        activity:
            The function that marks the active sites of an iteration, or `None` to count the sites that change.

    Attributes:
        values_:
            The record of the cascade of the last run, or `None` when the lattice did not change.

    See Also:
        DriveKernel: Runs one drive of a lattice state.
        RuleKernel: Runs a local rule of a lattice.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SandpileSystem
        >>> kernel = SandpileSystem().prepare().simulate(until=0).problem.lattice[0]
        >>> type(kernel).__name__, kernel.label
        ('CascadeKernel', 'avalanche')
    """

    values_: _CascadeRecordSpec | None

    def __init__(
        self,
        label: str,
        drive: tuple[RuleKernel | DriveKernel, ...],
        relax: tuple[RuleKernel | DriveKernel, ...],
        targets: tuple[_LatticeState, ...],
        stable: Callable[..., bool],
        *,
        max_iterations: int,
        activity: Callable[..., object] | None = None,
    ) -> None:
        self.label = label
        self.drive = drive
        self.relax = relax
        self.targets = targets
        self.stable = stable
        self.max_iterations = max_iterations
        self.activity = activity

    def _get_arrays(self, x: Array) -> list[Array]:
        """Return the targets' arrays in lattice shape."""
        return [np.reshape(x[state.sites], state.shape) for state in self.targets]

    def _is_stable(self, x: Array) -> bool:
        """Check the stability condition on the current state."""
        return bool(self.stable(*self._get_arrays(x)))

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Apply the drive, then relax until stable, and set the record of a cascade.

        A record is set only for a cascade that changed the lattice. The state vector of the state is updated in place,
        and the record holds the time of the state.

        Args:
            inputs:
                The state.
        """
        self.values_ = self._relax(self._get_state_of(inputs))

    def _relax(self, state: RunStateSpec) -> _CascadeRecordSpec | None:
        """Apply the drive, then relax until stable, and return the record of a cascade.

        A record is returned only for a cascade that changed the lattice.

        Args:
            state:
                The state.

        Returns:
            record:
                The record of the cascade, or `None` when the lattice did not change.

        Raises:
            SolverError:
                The lattice is not stable after the largest number of iterations.
        """
        x, time = state.x, state.t
        for law in self.drive:
            law.run(state)
        size = duration = 0
        changed: npt.NDArray[np.bool_] | None = None
        sites = np.concatenate([np.arange(state.start, state.start + state.size) for state in self.targets])
        while not self._is_stable(x):
            if duration >= self.max_iterations:
                error_msg = (
                    f'The cascade {self.label} is not stable after {self.max_iterations} iterations. Check its '
                    'stability condition or raise max_iterations.'
                )
                raise SolverError(error_msg)
            before = x[sites].copy()
            active = (
                np.ravel(np.asarray(self.activity(*self._get_arrays(x)), dtype=bool))
                if self.activity is not None
                else None
            )
            for law in self.relax:
                if active is not None and isinstance(law, RuleKernel) and law.schedule == 'synchronous':
                    law._run_near(state, active)
                else:
                    law.run(state)
            updated = (x[sites] != before).reshape(len(self.targets), -1).any(axis=0) if active is None else active
            size += int(updated.sum())
            changed = updated if changed is None else changed | updated
            duration += 1
        if not duration:
            return None
        area = int(changed.sum()) if changed is not None else 0
        return _CascadeRecordSpec(self.label, time, size, duration, area)
