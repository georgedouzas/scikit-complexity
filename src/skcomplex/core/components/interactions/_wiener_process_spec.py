"""Define the wiener process spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class WienerProcessSpec(BaseSpec):
    """A spec of a set of Wiener processes whose random steps are correlated.

    A Wiener process adds a random step of variance `h` over a time step `h`, and the steps of these processes have the
    covariance `h C`. An interaction takes the spec as an argument, such as `process=market`, and its noise laws name
    the steps as `s.process.dW`, or `s.process.dW[k]` for process `k` of several. Interactions that take processes of
    one label share them, so their laws move with the same random steps. Shared shocks move the prices of a market, a
    common current moves particles, and one weather pattern moves the yields of many farms.

    Read more in the user guide, under Laws.

    Args:
        label:
            The label of the processes, unique among the components of the system of the interactions that bind them.

        dimension:
            The number of processes.

        covariance:
            The constant covariance matrix `C` of the processes, symmetric and positive semidefinite of size
            `dimension`, or `None` for the identity.

    See Also:
        Interaction: Takes the spec as an argument.

    Examples:
        >>> from skcomplex.core.components.interactions import WienerProcessSpec
        >>> WienerProcessSpec('shared', dimension=2, covariance=((1.0, 0.5), (0.5, 1.0))).dimension
        2
    """

    label: str
    dimension: int = 1
    covariance: object = None

    def _check(self) -> None:
        """Raise an error for a label that is not a string or a dimension below 1.

        Raises:
            ValidationError:
                The label is not a string, or the dimension is not a positive integer.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of Wiener processes must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)
        if not (isinstance(self.dimension, int) and self.dimension >= 1):
            error_msg = f'Wiener processes have a positive whole dimension, but it is {self.dimension!r}.'
            raise ValidationError(error_msg)
