"""Define the draw log spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from ..types import Array
from ._base_spec import BaseSpec
from ._draw_record_spec import DrawRecordSpec


@dataclass(frozen=True)
class DrawLogSpec(BaseSpec):
    """Named draws and jump firings of one run, with their log-probability.

    A run records it with `simulate(record_draws=True)` and replays it with `simulate(replay=trajectory)`.

    Args:
        draws:
            The values of every named draw of the run, by the path of its owner and its name, such as
            `walker.kick`.

        firing_times:
            The time of every firing of a jump law, in order.

        firing_laws:
            The index of the jump law of every firing, in the order of the propensities of the problem.

        log_probability:
            The total log-probability of the draws and the firings under the parameters of the run.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import DrawLogSpec
        >>> DrawLogSpec({}, np.array([0.5]), np.array([1]), -1.5).firing_laws.tolist()
        [1]
    """

    draws: Mapping[str, DrawRecordSpec]
    firing_times: Array
    firing_laws: Array
    log_probability: float

    def _check(self) -> None:
        """Raise an error when firing times and laws differ in length or times decrease."""
        if len(self.firing_times) != len(self.firing_laws):
            error_msg = 'The firings hold times and laws of different lengths.'
            raise ValidationError(error_msg)
        if np.any(np.diff(self.firing_times) < 0):
            error_msg = 'The firing times must not decrease.'
            raise ValidationError(error_msg)
        for record in self.draws.values():
            record._check()
