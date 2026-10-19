"""Rank the parameters of a model by their effect on an output, one at a time, by Morris screening, or by Sobol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np
from SALib.analyze import morris as morris_analysis
from SALib.analyze import sobol as sobol_analysis
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

from ..base import ValidationError
from ._designs import Rows
from ._runner import Experiment, Output, Source

METHODS = ('oat', 'morris', 'sobol')


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity indices of an output with respect to parameters.

    Args:
        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        indices:
            A mapping from parameter to its indices: `effect` for one at a time, `mu_star`, `mu_star_conf`, and
            `sigma` for Morris, and `S1`, `S1_conf`, `ST`, and `ST_conf` for Sobol.

        runs:
            The number of model runs.
    """

    method: str
    indices: dict[str, dict[str, float]]
    runs: int

    def ranking(self) -> list[str]:
        """Return the parameters from the most to the least influential.

        Returns:
            parameters:
                The parameters, ranked by `effect`, `mu_star`, or `ST`.
        """
        key = {'oat': 'effect', 'morris': 'mu_star', 'sobol': 'ST'}[self.method]
        return sorted(self.indices, key=lambda name: -abs(self.indices[name][key]))


def _outputs(experiment: Experiment, output: Output) -> np.ndarray:
    """Return the scalar output of every run of an experiment."""
    result = experiment.run()
    name = output if isinstance(output, str) else getattr(output, '__name__', 'output')
    return np.asarray([run.scalars[name] for run in result.runs], dtype=float)


def sensitivity(
    source: Source,
    bounds: Mapping[str, tuple[float, float]],
    output: Output,
    *,
    method: str = 'sobol',
    n: int = 1024,
    until: float | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
    **settings: Any,  # noqa: ANN401
) -> SensitivityResult:
    """Return the sensitivity of a scalar output to parameters within bounds.

    One at a time moves each parameter from its lower to its upper bound with the others at their midpoints. Morris
    screening averages elementary effects over `n` trajectories. Sobol analysis decomposes the output's variance with
    Saltelli's sampling of `n` base points.

    Args:
        source:
            A registered model name, a factory, or a system.

        bounds:
            A mapping from factory argument or dotted path to its `(low, high)` bounds.

        output:
            The path of a state or observable, whose last value is the output, or a function of a trajectory that
            returns a scalar.

        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        n:
            The number of trajectories for Morris or of base points for Sobol, best a power of 2 for Sobol.

        until:
            The simulated time of every run, or `None` for the model's default.

        random_state:
            The seed of the sampling and of the runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

        **settings:
            Other `Experiment` arguments, such as `dt` or `solver`.

    Returns:
        result:
            The indices of every parameter.

    Raises:
        ValidationError:
            The method is unknown.

    Examples:
        >>> from skcomplex.core.experiments import sensitivity
        >>> from skcomplex.models.ecology import logistic_growth
        >>> result = sensitivity('logistic_growth', {'growth.r': (0.1, 1.0), 'growth.K': (50, 150)},
        ...                      'population.n', method='oat', until=20)
        >>> result.ranking()[0]
        'growth.K'
    """
    if method not in METHODS:
        error_msg = f'Unknown sensitivity method {method!r}. Use one of {", ".join(METHODS)}.'
        raise ValidationError(error_msg)
    names = list(bounds)
    problem = {'num_vars': len(names), 'names': names, 'bounds': [list(bounds[name]) for name in names]}

    def experiment(rows: list[dict[str, float]]) -> Experiment:
        return Experiment(
            source,
            Rows(rows),
            until=until,
            outputs=[output],
            random_state=random_state,
            executor=executor,
            **settings,
        )

    if method == 'oat':
        middle = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
        rows = [{**middle, name: bound} for name in names for bound in bounds[name]]
        values = _outputs(experiment(rows), output)
        indices = {
            name: {'effect': float(values[2 * index + 1] - values[2 * index])} for index, name in enumerate(names)
        }
        return SensitivityResult(method, indices, len(rows))
    if method == 'morris':
        samples = morris_sample.sample(problem, n, num_levels=4, seed=random_state)
        values = _outputs(experiment([dict(zip(names, map(float, row), strict=True)) for row in samples]), output)
        analysis = morris_analysis.analyze(problem, samples, values, seed=random_state)
        indices = {
            name: {
                'mu_star': float(analysis['mu_star'][index]),
                'mu_star_conf': float(analysis['mu_star_conf'][index]),
                'sigma': float(analysis['sigma'][index]),
            }
            for index, name in enumerate(names)
        }
        return SensitivityResult(method, indices, len(samples))
    samples = sobol_sample.sample(problem, n, calc_second_order=False, seed=random_state)
    values = _outputs(experiment([dict(zip(names, map(float, row), strict=True)) for row in samples]), output)
    analysis = sobol_analysis.analyze(problem, values, calc_second_order=False, seed=random_state)
    indices = {
        name: {key: float(analysis[key][index]) for key in ('S1', 'S1_conf', 'ST', 'ST_conf')}
        for index, name in enumerate(names)
    }
    return SensitivityResult(method, indices, len(samples))
