"""Define the critical exponent of a quantity near a continuous transition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....analysis import ScalingAnalysis


class CriticalExponentAnalysis(ScalingAnalysis):
    """Critical exponent of a quantity against the distance of a control parameter to its critical value.

    It is the scaling analysis of kind `'power'`, the slope of `log y` against `log |p - p_c|` on one side of the
    critical point, such as the exponent `beta` of an order parameter `m ~ |p - p_c|^beta` on its ordered side, 1/2 in
    mean-field theory.

    Args:
        variables:
            The variables of a trajectory to read, or `None` for every scalar series.

        parameters:
            The control parameter value of every row of the series.

        critical:
            The critical value of the control parameter.

        above:
            Whether the fit uses the values above the critical value, as for couplings, rather than below it, as for
            temperatures.

    Attributes:
        values_:
            The series, one per column.

        labels_:
            The label of every series.

        exponent_:
            The critical exponent of every series.

    Examples:
        >>> from skcomplex.models.physics.realizations import CriticalExponentAnalysis
        >>> exponent = CriticalExponentAnalysis(parameters=[0.9, 0.99], critical=1.0).check()
        >>> exponent.prepare([0.1**0.5, 0.01**0.5]).run().exponent_.round(6).tolist()
        [0.5]
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        parameters: Sequence[float] | None = None,
        critical: float | None = None,
        above: bool = False,
    ) -> None:
        super().__init__("power", variables, parameters=parameters, critical=critical, above=above)
