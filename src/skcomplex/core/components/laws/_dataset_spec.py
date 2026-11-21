"""Define the dataset spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...base import BaseSpec
from ...errors import ValidationError
from ...types import AnyArray

MIN_DIMENSIONS = 2


@dataclass(frozen=True)
class DatasetSpec(BaseSpec):
    """A spec of sequences of parameter values that a training run steps through.

    It gives the inputs a nested system reads at every step of its training, such as `DatasetSpec({'inputs.x': x})`
    with `x` of shape `(sequences, steps, 1)` for a recurrent network that reads one number per step. At step `k` of
    sequence `i` every parameter takes its value at `[i, k]` before the laws of the step run, as a schedule sets a
    parameter in a simulation, and the loss reads the values the step gives. Every sequence starts from the initial
    values of the system. It holds the words of sentences for a language model, the daily prices for a forecast, and
    the stimuli of a neural circuit.

    Read more in the user guide, under Laws.

    Args:
        values:
            A mapping from the dotted path of a parameter of the system, such as `inputs.x`, to its values.

        weights:
            The weights of the steps in the loss and in every metric, an array of shape `(sequences, steps)`, a
            mapping from the label of the loss or of a metric to such an array, the others weighting every step
            equally, or `None` for equal weights.

        times:
            The times of the ends of the inner steps of a system of ordinary differential equations, one per step and
            increasing from a positive time, since the run starts at time 0, or `None` for the times `1, 2, ...`.

    Raises:
        ValidationError:
            The values are empty, a path is not a string, or an array has fewer than two dimensions.

    See Also:
        TrainingInteraction: Trains the parameters of a nested system on the spec.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.laws import DatasetSpec
        >>> data = DatasetSpec({'inputs.x': np.zeros((5, 12, 1))})
        >>> data.sequences, data.steps
        (5, 12)
    """

    values: Mapping[str, AnyArray]
    weights: AnyArray | Mapping[str, AnyArray] | None = None
    times: AnyArray | None = None

    def _check(self) -> None:
        """Raise an error for empty values, a path that is no string, or a flat array.

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
        """Number of inner steps of every sequence.

        It is the length of the second axis of the first array.
        """
        return int(np.shape(next(iter(self.values.values())))[1])
