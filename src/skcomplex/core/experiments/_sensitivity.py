"""Rank the inputs of an experiment by their effect on an output, one at a time, by Morris screening, or by Sobol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
from SALib.analyze import morris as morris_analysis
from SALib.analyze import sobol as sobol_analysis

from ..errors import AnalysisError
from ..types import Array
from ._results import ExperimentResult

METHODS = ('oat', 'morris', 'sobol')


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity of a scalar output of an experiment to the values of its design.

    Args:
        names:
            The inputs, the names of the design values.

        inputs:
            The design values of every run, one column per input.

        outputs:
            The output of every run.

        ranking:
            The inputs from the most to the least influential, by `effects`, `mu_star`, or `total_order`.

        effects:
            The one-at-a-time effect of every input, for `'oat'`.

        mu_star:
            The mean absolute elementary effect of every input, for `'morris'`.

        mu_star_conf:
            The confidence interval of `mu_star`, for `'morris'`.

        sigma:
            The standard deviation of the elementary effects of every input, for `'morris'`.

        first_order:
            The first-order Sobol index of every input, for `'sobol'`.

        first_order_conf:
            The confidence interval of `first_order`, for `'sobol'`.

        total_order:
            The total Sobol index of every input, for `'sobol'`.

        total_order_conf:
            The confidence interval of `total_order`, for `'sobol'`.
    """

    names: list[str]
    inputs: Array
    outputs: Array
    ranking: list[str]
    effects: dict[str, float] | None = None
    mu_star: dict[str, float] | None = None
    mu_star_conf: dict[str, float] | None = None
    sigma: dict[str, float] | None = None
    first_order: dict[str, float] | None = None
    first_order_conf: dict[str, float] | None = None
    total_order: dict[str, float] | None = None
    total_order_conf: dict[str, float] | None = None


def sensitivity(
    source: ExperimentResult,
    *,
    method: str = 'sobol',
    output: str | None = None,
    random_state: int | None = None,
) -> SensitivityResult:
    """Return the sensitivity of a scalar output of an experiment to the values of its design, with SALib.

    The source is the result of an experiment whose design matches the method, with one run per row:

    - `'oat'` reads a `OneAtATimeDesign` and gives the change of the output from the lower to the upper bound of every
      input, with the others at their midpoints.
    - `'morris'` reads a `MorrisDesign` and gives the Morris elementary effects.
    - `'sobol'` reads a `SaltelliDesign` and gives the first-order and total Sobol indices.

    Args:
        source:
            The result of the experiment.

        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        output:
            The name of the scalar output, such as the path of a kept series, whose last value it is.

        random_state:
            The seed of the bootstrap confidence intervals.

    Returns:
        result:
            The design values, the outputs, the indices of the method, and the ranking of the inputs.

    Raises:
        AnalysisError:
            The method is unknown, the output is missing, or the source is not the result of an experiment with one
            run per design row.

    Examples:
        >>> from skcomplex.core.experiments import Experiment, OneAtATimeDesign, sensitivity
        >>> from skcomplex.systems.biology.ecology import logistic_growth
        >>> bounds = {'growth.r': (0.1, 1.0), 'growth.K': (50, 150)}
        >>> experiment = Experiment(logistic_growth, OneAtATimeDesign(), bounds, until=20, outputs=['population.n'])
        >>> result = experiment.prepare().run().result_
        >>> sensitivity(result, method='oat', output='population.n').ranking[0]
        'growth.K'
    """
    if method not in METHODS:
        error_msg = f'Unknown sensitivity method {method!r}. Use one of {", ".join(METHODS)}.'
        raise AnalysisError(error_msg)
    if not isinstance(output, str):
        error_msg = f'output = {output!r} must be the name of a scalar output of the experiment.'
        raise AnalysisError(error_msg)
    if not isinstance(source, ExperimentResult) or not source.runs:
        error_msg = f'A sensitivity analysis reads the result of an experiment with runs, not {source!r}.'
        raise AnalysisError(error_msg)
    if any(run.replicate or run.scenario != source.runs[0].scenario for run in source.runs):
        error_msg = 'A sensitivity analysis reads one run per design row, with one scenario and one replicate.'
        raise AnalysisError(error_msg)
    names = list(source.runs[0].row)
    inputs = np.asarray([[run.row[name] for name in names] for run in source.runs], dtype=float)
    outputs = np.asarray(source.column(output), dtype=float)
    problem = {'num_vars': len(names), 'names': names}
    if method == 'oat':
        effects = dict(zip(names, map(float, outputs[1::2] - outputs[0::2]), strict=True))
        return SensitivityResult(names, inputs, outputs, _rank(names, effects), effects=effects)
    if method == 'morris':
        indices = morris_analysis.analyze(problem, inputs, outputs, seed=random_state)
        mu_star = _by_name(names, indices['mu_star'])
        return SensitivityResult(
            names,
            inputs,
            outputs,
            _rank(names, mu_star),
            mu_star=mu_star,
            mu_star_conf=_by_name(names, indices['mu_star_conf']),
            sigma=_by_name(names, indices['sigma']),
        )
    indices = sobol_analysis.analyze(problem, outputs, calc_second_order=False, seed=random_state)
    total_order = _by_name(names, indices['ST'])
    return SensitivityResult(
        names,
        inputs,
        outputs,
        _rank(names, total_order),
        first_order=_by_name(names, indices['S1']),
        first_order_conf=_by_name(names, indices['S1_conf']),
        total_order=total_order,
        total_order_conf=_by_name(names, indices['ST_conf']),
    )


def _by_name(names: list[str], values: Array) -> dict[str, float]:
    """Return the values of the inputs by name."""
    return dict(zip(names, map(float, values), strict=True))


def _rank(names: list[str], values: dict[str, float]) -> list[str]:
    """Return the inputs from the largest to the smallest absolute value."""
    return sorted(names, key=lambda name: -abs(values[name]))
