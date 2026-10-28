"""Define the phases schedule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from ..errors import ValidationError
from ._phase_spec import PhaseSpec
from ._schedule_spec import ScheduleSpec

ACTIVATIONS = ('random', 'sequential', 'simultaneous', 'by-type')


@dataclass(frozen=True)
class PhasesScheduleSpec(ScheduleSpec):
    """Spec of a schedule of discrete time that runs named phases in order in every step.

    Args:
        phases:
            The phases, each with its rules and activation policy.

    Examples:
        >>> from skcomplex.specs import PhaseSpec, PhasesScheduleSpec
        >>> [phase.name for phase in PhasesScheduleSpec((PhaseSpec('produce'), PhaseSpec('trade'))).check().phases]
        ['produce', 'trade']
    """

    phases: Sequence[PhaseSpec]

    def _check(self) -> Self:
        """Raise an error for a phase with an unknown activation or no round, or for a repeated name.

        Returns:
            spec:
                The spec itself.

        Raises:
            ValidationError:
                A phase repeats the name of another, has an unknown activation, or runs no round.
        """
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            error_msg = f'The phases {names} repeat a name. Give every phase of a step its own name.'
            raise ValidationError(error_msg)
        for phase in self.phases:
            if phase.activation not in ACTIVATIONS:
                error_msg = (
                    f'Unknown activation {phase.activation!r} of phase {phase.name!r}. Use one of {ACTIVATIONS}.'
                )
                raise ValidationError(error_msg)
            if phase.repeat < 1:
                error_msg = f'Phase {phase.name!r} repeats {phase.repeat} times. Use at least one round.'
                raise ValidationError(error_msg)
        return self
