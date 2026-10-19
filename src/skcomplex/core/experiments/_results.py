"""Hold the results of experiments as tidy rows, kept series, and summaries across replicates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RunResult:
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
            The scalar outputs by name: the last value of every kept scalar series and the values of output functions.
    """

    index: int
    row: dict[str, Any]
    scenario: str
    replicate: int
    seed: int
    times: npt.NDArray[np.float64]
    series: dict[str, npt.NDArray[Any]]
    scalars: dict[str, Any]


class ExperimentResult:
    """Results of an experiment: one row per run, the kept series, and summaries across replicates.

    Args:
        runs:
            The results of every run, in the experiment's order.

    Examples:
        >>> from skcomplex.core.experiments import Experiment, Grid
        >>> from skcomplex.models.biology.epidemiology import sir
        >>> result = Experiment('sir', Grid(gamma=[0.1, 0.2]), until=10).run()
        >>> [row['gamma'] for row in result.rows]
        [0.1, 0.2]
    """

    def __init__(self, runs: Sequence[RunResult]) -> None:
        self.runs = list(runs)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """One row per run with its design values, scenario, replicate, seed, and scalar outputs."""
        return [
            {**run.row, 'scenario': run.scenario, 'replicate': run.replicate, 'seed': run.seed, **run.scalars}
            for run in self.runs
        ]

    def column(self, name: str) -> npt.NDArray[Any]:
        """Return one column of the rows.

        Args:
            name:
                The name of a design value, `scenario`, `replicate`, `seed`, or a scalar output.

        Returns:
            values:
                One value per run.
        """
        return np.asarray([row[name] for row in self.rows])

    def series(self, name: str) -> npt.NDArray[Any]:
        """Return a kept series of every run, stacked when all runs have the same number of frames.

        Args:
            name:
                The path of the kept series.

        Returns:
            series:
                One row per run, or an object array of per-run series of different lengths.
        """
        values = [run.series[name] for run in self.runs]
        if len({value.shape for value in values}) == 1:
            return np.stack(values)
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    def summary(self, quantiles: Sequence[float] = (0.05, 0.5, 0.95)) -> list[dict[str, Any]]:
        """Return the mean, standard deviation, and quantiles of every scalar output per design row and scenario.

        Args:
            quantiles:
                The quantiles to report.

        Returns:
            summary:
                One row per design row and scenario, with `<output>.mean`, `<output>.std`, and `<output>.q<quantile>`
                columns and the number of replicates as `runs`.
        """
        groups: dict[tuple[Any, ...], list[RunResult]] = defaultdict(list)
        for run in self.runs:
            groups[(tuple(run.row.items()), run.scenario)].append(run)
        summary = []
        for (row, scenario), runs in groups.items():
            entry: dict[str, Any] = {**dict(row), 'scenario': scenario, 'runs': len(runs)}
            for name in runs[0].scalars:
                values = np.asarray([run.scalars[name] for run in runs], dtype=float)
                entry[f'{name}.mean'] = float(values.mean())
                entry[f'{name}.std'] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                for quantile in quantiles:
                    entry[f'{name}.q{quantile:g}'] = float(np.quantile(values, quantile))
            summary.append(entry)
        return summary

    def to_frame(self) -> Any:  # noqa: ANN401
        """Return the rows as a pandas data frame.

        Returns:
            frame:
                The data frame.

        Raises:
            ImportError:
                pandas is not installed.
        """
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as error:
            error_msg = 'to_frame needs pandas. Install it, or use `rows`, which needs nothing more.'
            raise ImportError(error_msg) from error
        return pd.DataFrame(self.rows)
