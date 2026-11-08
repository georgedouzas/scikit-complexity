"""Hold the results of experiments.

They are tidy rows, kept series, ensembles, and summaries across replicates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..specs._run_result_spec import _RunResultSpec


@dataclass(frozen=True)
class ExperimentResult:
    """Results of an experiment.

    They hold one row per run, the kept series, ensembles, and summaries across replicates. `sweep`, `ensemble`, and
    `compare_scenarios` return it. Its rows form a tidy table, one row per run and one column per value of the run,
    scenario, replicate, seed, and scalar output, which `to_frame` returns as a data frame.

    Args:
        runs:
            The results of every run, in the experiment's order.

    Examples:
        >>> from skcomplex.core.experiments import sweep
        >>> from skcomplex.systems.biology.epidemiology import SIRSystem
        >>> result = sweep(SIRSystem(), {'gamma': [0.1, 0.2]}, until=10)
        >>> [row['gamma'] for row in result.rows]
        [0.1, 0.2]
    """

    runs: tuple[_RunResultSpec, ...]

    @property
    def rows(self) -> list[dict[str, object]]:
        """One row per run with its values and outputs.

        A row holds its design values, scenario, replicate, seed, and scalar outputs.
        """
        return [
            {**run.row, 'scenario': run.scenario, 'replicate': run.replicate, 'seed': run.seed, **run.scalars}
            for run in self.runs
        ]

    def column(self, name: str) -> npt.NDArray[np.generic]:
        """Return one column of the rows.

        Args:
            name:
                The name of a design value, `scenario`, `replicate`, `seed`, or a scalar output.

        Returns:
            values:
                One value per run.
        """
        return np.asarray([row[name] for row in self.rows])

    def series(self, name: str) -> npt.NDArray[np.generic]:
        """Return a kept series of every run.

        The series are stacked when all runs have the same number of frames.

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

    def ensemble(
        self,
        name: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Return a kept series of every run on common times, with the seeds of the runs.

        The series are interpolated on the times of the shortest run.

        Args:
            name:
                The path of the kept series.

        Returns:
            times:
                The common times.

            values:
                One row per run and one column per common time.

            seeds:
                The seed of the run of every row.

        Examples:
            >>> from skcomplex.core.experiments import ensemble
            >>> from skcomplex.systems.biology.epidemiology import SIRSystem
            >>> result = ensemble(SIRSystem(stochastic=True), replicates=3, until=10, random_state=0)
            >>> times, values, seeds = result.ensemble('I.n')
            >>> values.shape == (3, times.size), seeds.tolist()
            (True, [0, 1, 2])
        """
        times = min((np.asarray(run.times, dtype=float) for run in self.runs), key=len)
        values = np.asarray(
            [
                np.interp(times, np.asarray(run.times, dtype=float), np.asarray(run.series[name], dtype=float))
                for run in self.runs
            ],
        )
        seeds = np.asarray([run.seed for run in self.runs], dtype=np.int64)
        return times, values, seeds

    def summary(self, quantiles: Sequence[float] = (0.05, 0.5, 0.95)) -> list[dict[str, object]]:
        """Return statistics of every numeric scalar output per row and scenario.

        They are the mean, the standard deviation, and the quantiles.

        Args:
            quantiles:
                The quantiles to report.

        Returns:
            summary:
                One row per design row and scenario, with `<output>.mean`, `<output>.std`, and `<output>.q<quantile>`
                columns and the number of replicates as `runs`.
        """
        groups: dict[tuple[tuple[tuple[str, object], ...], str], list[_RunResultSpec]] = defaultdict(list)
        for run in self.runs:
            groups[(tuple(run.row.items()), run.scenario)].append(run)
        summary = []
        for (row, scenario), runs in groups.items():
            entry: dict[str, object] = {**dict(row), 'scenario': scenario, 'runs': len(runs)}
            for name in runs[0].scalars:
                values = np.asarray([run.scalars[name] for run in runs])
                if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
                    continue
                values = values.astype(float)
                entry[f'{name}.mean'] = float(values.mean())
                entry[f'{name}.std'] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                for quantile in quantiles:
                    entry[f'{name}.q{quantile:g}'] = float(np.quantile(values, quantile))
            summary.append(entry)
        return summary

    def to_frame(self) -> pd.DataFrame:
        """Return the rows as a pandas data frame.

        Returns:
            frame:
                The data frame.
        """
        return pd.DataFrame(self.rows)
