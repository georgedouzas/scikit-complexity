"""Define the wiener process spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class WienerProcessSpec(BaseSpec):
    """Spec of a set of Wiener processes of correlated increments.

    Their increments over a step `h` have the covariance `h C`. An interaction binds the processes through a role
    declared with `RoleSpec(noise=True)`, and its noise laws name them through the handle `namespace.<role>.dW`, a
    symbol for one process and a vector of symbols for several, so `namespace.<role>.dW[k]` names process `k`.
    Interactions of one system that bind processes of one label share them, so laws that name one process share its
    increment.

    Args:
        label:
            The label of the processes, unique among the parts of the system of the interactions that bind them.

        dimension:
            The number of processes.

        covariance:
            The constant covariance matrix `C` of the processes, symmetric and positive semidefinite of size
            `dimension`, or `None` for the identity.

    Examples:
        >>> from skcomplex.core.specs import WienerProcessSpec
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
