"""Define the Binder cumulant of samples of an order parameter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import DistributionAnalysis


class BinderCumulantAnalysis(DistributionAnalysis):
    """Binder cumulant `1 - <m^4> / (3 <m^2>^2)` of every series of samples of an order parameter.

    It is the distribution analysis of kind `'moment-ratio'` with the orders 4 and 2. It is 2/3 deep in an ordered
    phase and 0 for Gaussian fluctuations about 0, and the cumulants of several system sizes cross at the critical
    point, which `CollapseAnalysis('crossings')` locates.

    Args:
        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        group:
            The path of the agent or particle group whose speeds are read, or `None` for the series of a trajectory.

    Attributes:
        values_:
            The samples, one per column.

        labels_:
            The label of every series.

        ratio_:
            The ratio `<m^4> / <m^2>^2` of every series.

        cumulant_:
            The Binder cumulant of every series.

    Examples:
        >>> from skcomplex.models.physics.realizations import BinderCumulantAnalysis
        >>> BinderCumulantAnalysis().check().prepare([1.0, -1.0, 1.0]).run().cumulant_.round(4).tolist()
        [0.6667]
    """

    def __init__(self, variables: Sequence[str] | None = None, *, group: str | None = None) -> None:
        super().__init__("moment-ratio", variables, group=group, orders=(4, 2))
