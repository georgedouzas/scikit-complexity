"""Define the sampler spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...enums import SamplerKind, SweepKind
from ...errors import ValidationError


@dataclass(frozen=True)
class SamplerSpec(BaseSpec):
    """A spec of the rule that keeps or rejects the moves of a Monte Carlo sweep.

    It decides whether a proposed move of a site is kept, from the change `d` of the cost of the site at an intensity,
    such as `SamplerSpec('metropolis', s.beta)` for a magnet at the inverse temperature `beta`, which always keeps a
    move that lowers the energy and keeps one that raises it with probability `exp(-beta d)`. It sweeps the lattice
    once per unit of time, in the order of `schedule`. It samples the spins of a magnet, the letters of a sequence of
    DNA, and the opinions of a population at a level of noise.

    Read more in the user guide, under Laws.

    Args:
        kind:
            `'metropolis'` to accept a change `d` of the objective with probability `min(1, exp(-intensity * d))`,
            `'glauber'` with `1 / (1 + exp(intensity * d))`, or `'heat-bath'` to draw the new value among all values of
            a discrete move with weights `exp(-intensity * objective)`.

        intensity:
            The intensity, usually a parameter symbol so that it changes live.

        schedule:
            `'checkerboard'` to update the two halves of a square lattice in turn, like the squares of a chessboard,
            `'colored'` to update groups of sites with no neighbours in common in turn, or `'random-sequential'` to
            visit the sites one at a time in a random order, or `'auto'` for the checkerboard when the lattice allows
            it, the colored schedule for sums over cofaces, and the random order otherwise.

    See Also:
        MoveSpec: The change of a site that a sweep proposes.

    Examples:
        >>> from skcomplex.core.components.laws import SamplerSpec
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
        if self.kind not in SamplerKind:
            error_msg = f'Unknown sampler {self.kind!r}. Use one of {", ".join(SamplerKind)}.'
            raise ValidationError(error_msg)
        if self.schedule not in SweepKind:
            error_msg = f'Unknown Monte Carlo schedule {self.schedule!r}. Use one of {", ".join(SweepKind)}.'
            raise ValidationError(error_msg)
