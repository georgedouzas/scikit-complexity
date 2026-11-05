"""Define the dataset spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from ..types import AnyArray
from ._base_spec import BaseSpec

MIN_DIMENSIONS = 2


@dataclass(frozen=True)
class DatasetSpec(BaseSpec):
    """Sequences of values of the parameters of a model, one value per sequence and inner step, with step weights.

    Every array has the shape `(sequences, steps)` followed by the shape of its parameter. At inner step `k` of
    sequence `i` every parameter takes its value at `[i, k]` before the laws of the step are evaluated, as a schedule
    sets a parameter in a simulation, and the loss reads the states the step gives. Every sequence starts from the
    initial states of the model.

    Args:
        values:
            A mapping from the dotted path of a parameter of the model, such as `inputs.x`, to its values.

        weights:
            The weights of the steps in the loss and in every metric, an array of shape `(sequences, steps)`, a
            mapping from the label of the loss or of a metric to such an array, the others weighting every step
            equally, or `None` for equal weights.

        times:
            The times of the ends of the inner steps of a model of ordinary differential equations, one per step and
            increasing from a positive time, since the run starts at time 0, or `None` for the times `1, 2, ...`.

    Raises:
        ValidationError:
            The values are empty, a path is not a string, or an array has fewer than two dimensions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.specs import DatasetSpec
        >>> data = DatasetSpec({'inputs.x': np.zeros((5, 12, 1))})
        >>> data.sequences, data.steps
        (5, 12)
    """

    values: Mapping[str, AnyArray]
    weights: AnyArray | Mapping[str, AnyArray] | None = None
    times: AnyArray | None = None

    def _check(self) -> None:
        """Raise an error for empty values, a path that is not a string, or an array with fewer than two dimensions.

        Raises:
            ValidationError:
                The values are empty, a path is not a string, or an array has fewer than two dimensions.
        """
        if not isinstance(self.values, Mapping) or not self.values:
            error_msg = 'A dataset maps the dotted paths of parameters to arrays. Give at least one array.'
            raise ValidationError(error_msg)
        for path, array in self.values.items():
            if not isinstance(path, str):
                error_msg = f'A dataset maps dotted paths of parameters to arrays, not {path!r}.'
                raise ValidationError(error_msg)
            if np.ndim(array) < MIN_DIMENSIONS:
                error_msg = (
                    f'The values of {path} have the shape {np.shape(array)}. Give an array of shape '
                    '(sequences, steps) followed by the shape of the parameter.'
                )
                raise ValidationError(error_msg)

    @property
    def sequences(self) -> int:
        """Number of sequences, the length of the first axis of the first array."""
        return int(np.shape(next(iter(self.values.values())))[0])

    @property
    def steps(self) -> int:
        """Number of inner steps of every sequence, the length of the second axis of the first array."""
        return int(np.shape(next(iter(self.values.values())))[1])
