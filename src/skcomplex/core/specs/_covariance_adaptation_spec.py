"""Define the settings of covariance matrix adaptation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class CovarianceAdaptationSpec(BaseSpec):
    """Settings of covariance matrix adaptation.

    Args:
        sigma:
            The initial step size, relative to the box of the candidates.

    Examples:
        >>> from skcomplex.core.specs import CovarianceAdaptationSpec
        >>> CovarianceAdaptationSpec().sigma
        0.1
    """

    sigma: float = 0.1
