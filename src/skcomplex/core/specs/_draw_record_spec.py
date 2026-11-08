"""Define the draw record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ..errors import ValidationError
from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class DrawRecordSpec(BaseSpec):
    """Values a run drew under one name, with the arguments of their distribution.

    Args:
        kind:
            `'uniform'`, `'normal'`, `'bernoulli'`, `'categorical'`, or `'draw'` for a distribution of `scipy.stats`.

        distribution:
            The name of the `scipy.stats` distribution of a draw of kind `'draw'`, or `None`.

        steps:
            The step of the run every value was drawn in.

        values:
            The drawn values, in the order the run drew them.

        arguments:
            The value of every argument of the distribution per drawn value, by keyword, such as `mean` and
            `deviation` of a normal draw, `low` and `high` of a uniform draw, `p` of a Bernoulli draw, `weights` of a
            categorical draw with one row per value, and the keywords of a draw from `scipy.stats`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import DrawRecordSpec
        >>> p = {'p': np.array([0.3, 0.3])}
        >>> record = DrawRecordSpec('bernoulli', None, np.array([1, 2]), np.array([1.0, 0.0]), p)
        >>> record.values.tolist()
        [1.0, 0.0]
    """

    kind: str
    distribution: str | None
    steps: Array
    values: Array
    arguments: Mapping[str, Array]

    def _check(self) -> None:
        """Raise an error when the steps, the values, and the arguments differ in length."""
        lengths = {len(self.steps), len(self.values), *(len(item) for item in self.arguments.values())}
        if len(lengths) > 1:
            error_msg = f'The draws of kind {self.kind} hold steps, values, and arguments of different lengths.'
            raise ValidationError(error_msg)
