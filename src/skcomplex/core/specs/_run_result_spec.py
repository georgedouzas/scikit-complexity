"""Define the run result spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _RunResultSpec(BaseSpec):
    """Result of one run of an experiment.

    Args:
        index:
            The position of the run in the experiment.

        row:
            The design values.

        scenario:
            The name of the scenario.

        replicate:
            The replicate number.

        seed:
            The index of the run's random stream among the streams of the experiment.

        times:
            The times of the kept series.

        series:
            The kept series by path.

        scalars:
            The scalar outputs by name: the last value of every kept scalar series, the values of output functions,
            and the results of the analyses.
    """

    index: int
    row: dict[str, object]
    scenario: str
    replicate: int
    seed: int
    times: npt.NDArray[np.float64]
    series: dict[str, npt.NDArray[np.generic]]
    scalars: dict[str, object]
