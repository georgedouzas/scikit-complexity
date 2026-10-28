"""Rank the inputs of an experiment by their effect on an output, one at a time, by Morris screening, or by Sobol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np
from SALib.analyze import morris as morris_analysis
from SALib.analyze import sobol as sobol_analysis

from ..analysis._base import BaseAnalysis
from ..analysis._kinds import DataKind
from ..errors import AnalysisError
from ._results import ExperimentResult

METHODS = ('oat', 'morris', 'sobol')


class SensitivityAnalysis(BaseAnalysis):
    """Sensitivity of a scalar output of an experiment to the values of its design, with SALib.

    The analysis prepares on the result of an experiment whose design matches the method, with one run per row:

    - `'oat'` reads a `OneAtATimeDesign` and gives the change of the output from the lower to the upper bound of every
      input, with the others at their midpoints.
    - `'morris'` reads a `MorrisDesign` and gives the Morris elementary effects.
    - `'sobol'` reads a `SaltelliDesign` and gives the first-order and total Sobol indices.

    Args:
        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        output:
            The name of the scalar output, such as the path of a kept series, whose last value it is.

        random_state:
            The seed of the bootstrap confidence intervals.

    Attributes:
        names_:
            The inputs, the names of the design values.

        inputs_:
            The design values of every run, one column per input.

        outputs_:
            The output of every run.

        effects_:
            The one-at-a-time effect of every input, for `'oat'`.

        mu_star_, mu_star_conf_, sigma_:
            The mean absolute elementary effect, its confidence interval, and the standard deviation of the elementary
            effects of every input, for `'morris'`.

        first_order_, first_order_conf_, total_order_, total_order_conf_:
            The first-order and total Sobol indices of every input and their confidence intervals, for `'sobol'`.

        ranking_:
            The inputs from the most to the least influential, by `effects_`, `mu_star_`, or `total_order_`.

    Examples:
        >>> from skcomplex.experiments import Experiment, OneAtATimeDesign, SensitivityAnalysis
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> bounds = {'growth.r': (0.1, 1.0), 'growth.K': (50, 150)}
        >>> experiment = Experiment('logistic_growth', OneAtATimeDesign(), bounds, until=20, outputs=['population.n'])
        >>> result = experiment.check().prepare().run().result_
        >>> SensitivityAnalysis('oat', 'population.n').check().prepare(result).run().ranking_[0]
        'growth.K'
    """

    def __init__(self, method: str = 'sobol', output: str | None = None, random_state: int | None = None) -> None:
        self.method = method
        self.output = output
        self.random_state = random_state

    def _check(self) -> Self:
        """Raise an error for an unknown method or a missing output."""
        if self.method not in METHODS:
            error_msg = f'Unknown sensitivity method {self.method!r}. Use one of {", ".join(METHODS)}.'
            raise AnalysisError(error_msg)
        if not isinstance(self.output, str):
            error_msg = f'output = {self.output!r} must be the name of a scalar output of the experiment.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind, since the analysis reads the runs of an experiment."""
        return DataKind.ENSEMBLE

    def _prepare(self, source: ExperimentResult) -> Self:  # type: ignore[override]
        """Read the design values and the output of every run of an experiment."""
        if not isinstance(source, ExperimentResult) or not source.runs:
            error_msg = f'A sensitivity analysis reads the result of an experiment with runs, not {source!r}.'
            raise AnalysisError(error_msg)
        if any(run.replicate or run.scenario != source.runs[0].scenario for run in source.runs):
            error_msg = 'A sensitivity analysis reads one run per design row, with one scenario and one replicate.'
            raise AnalysisError(error_msg)
        self.names_ = list(source.runs[0].row)
        self.inputs_ = np.asarray([[run.row[name] for name in self.names_] for run in source.runs], dtype=float)
        self.outputs_ = np.asarray(source.column(self.output), dtype=float)  # type: ignore[arg-type]
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the indices of the method and rank the inputs."""
        problem = {'num_vars': len(self.names_), 'names': self.names_}
        if self.method == 'oat':
            effects = self.outputs_[1::2] - self.outputs_[0::2]
            self.effects_ = dict(zip(self.names_, map(float, effects), strict=True))
            ranked = self.effects_
        elif self.method == 'morris':
            indices = morris_analysis.analyze(problem, self.inputs_, self.outputs_, seed=self.random_state)
            self.mu_star_ = dict(zip(self.names_, map(float, indices['mu_star']), strict=True))
            self.mu_star_conf_ = dict(zip(self.names_, map(float, indices['mu_star_conf']), strict=True))
            self.sigma_ = dict(zip(self.names_, map(float, indices['sigma']), strict=True))
            ranked = self.mu_star_
        else:
            indices = sobol_analysis.analyze(problem, self.outputs_, calc_second_order=False, seed=self.random_state)
            self.first_order_ = dict(zip(self.names_, map(float, indices['S1']), strict=True))
            self.first_order_conf_ = dict(zip(self.names_, map(float, indices['S1_conf']), strict=True))
            self.total_order_ = dict(zip(self.names_, map(float, indices['ST']), strict=True))
            self.total_order_conf_ = dict(zip(self.names_, map(float, indices['ST_conf']), strict=True))
            ranked = self.total_order_
        self.ranking_ = sorted(self.names_, key=lambda name: -abs(ranked[name]))
        return self
