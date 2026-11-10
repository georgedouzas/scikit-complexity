"""Define the sampler spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

SAMPLERS = ('metropolis', 'glauber', 'heat-bath')
MONTE_CARLO_SCHEDULES = ('checkerboard', 'random-sequential', 'colored')


@dataclass(frozen=True)
class SamplerSpec(BaseSpec):
    """A spec of the acceptance of Monte Carlo moves at an intensity.

    It makes one sweep of the lattice per unit of time.

    Args:
        kind:
            `'metropolis'` to accept a change `d` of the objective with probability `min(1, exp(-intensity * d))`,
            `'glauber'` with `1 / (1 + exp(intensity * d))`, or `'heat-bath'` to draw the new value among all values of
            a discrete move with weights `exp(-intensity * objective)`.

        intensity:
            The intensity, usually a parameter symbol so that it changes live.

        schedule:
            `'checkerboard'` to update the two sublattices of a square lattice in turn, `'colored'` to update the
            classes of a proper coloring of the neighbour graph in turn, or `'random-sequential'` to visit the sites one
            at a time in a random order.

    Examples:
        >>> from skcomplex.core.specs import SamplerSpec
        >>> SamplerSpec('glauber', 2.0).schedule
        'checkerboard'
    """

    kind: str = 'metropolis'
    intensity: object = 1.0
    schedule: str = 'checkerboard'

    def _check(self) -> None:
        """Raise an error for an unknown kind of sampler or schedule.

        Raises:
            ValidationError:
                The kind of sampler or the schedule is unknown.
        """
        if self.kind not in SAMPLERS:
            error_msg = f'Unknown sampler {self.kind!r}. Use one of {", ".join(SAMPLERS)}.'
            raise ValidationError(error_msg)
        if self.schedule not in MONTE_CARLO_SCHEDULES:
            error_msg = (
                f'Unknown Monte Carlo schedule {self.schedule!r}. Use one of {", ".join(MONTE_CARLO_SCHEDULES)}.'
            )
            raise ValidationError(error_msg)
