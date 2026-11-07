"""Rank the inputs of a system by their effect on an output, one at a time, by Morris screening, or by Sobol."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
from SALib.analyze import morris as morris_analysis
from SALib.analyze import sobol as sobol_analysis

from ..components.specs import ScenarioSpec
from ..engine.solvers import BaseSolver
from ..errors import AnalysisError
from ..specs import ObservableSpec
from ..types import Array
from ._designs import _get_sensitivity_rows
from ._runs import _run_experiment
from .types import Bounds, Source

METHODS = ('oat', 'morris', 'sobol')
SIZES = {'oat': 0, 'morris': 10, 'sobol': 1024}
"""The default number of Morris trajectories or of base points of every method."""


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity of a scalar output of a system to its inputs.

    Args:
        names:
            The inputs, model parameters or dotted paths.

        inputs:
            The values of the inputs of every run, one column per input.

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
    system: Source,
    bounds: Bounds,
    *,
    output: str,
    method: str = 'sobol',
    n: int | None = None,
    until: float | None = None,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    observables: Sequence[ObservableSpec] = (),
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> SensitivityResult:
    """Return the sensitivity of a scalar output of a system to its inputs within bounds, with SALib.

    The function runs the system once per row of the design the method reads, as `sweep` does, and analyses the
    output of the runs:

    - `'oat'` moves every input from its lower to its upper bound with the others at their midpoints, and gives the
      change of the output.
    - `'morris'` runs `n` Morris trajectories on four levels, 10 by default, and gives the Morris elementary effects.
    - `'sobol'` runs Saltelli's sample of `n` base points, 1024 by default and best a power of 2, and gives the
      first-order and total Sobol indices.

    Args:
        system:
            The system, whose copies take model parameters through `set_params`, or a factory of systems, such as a
            system class, which takes them as keyword arguments.

        bounds:
            A pair `(low, high)` per input, a model parameter or a dotted path of a state or a parameter of a part.

        output:
            The state or observable path whose last value is the scalar output.

        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        n:
            The number of Morris trajectories or of base points of Saltelli's sample, or `None` for the default.

        until:
            The end time of every run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        observables:
            The observables of every run, or none for the tuned ones.

        random_state:
            The seed of the design, of the runs, and of the bootstrap confidence intervals.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor.

    Returns:
        result:
            The inputs of every run, the outputs, the indices of the method, and the ranking of the inputs.

    Raises:
        AnalysisError:
            The method is unknown or the output is not a name.

        ValidationError:
            A bound is not a pair, or an input is no model parameter, state, or parameter.

    Examples:
        >>> from skcomplex.core.experiments import sensitivity
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> bounds = {'growth.r': (0.1, 1.0), 'growth.K': (50, 150)}
        >>> result = sensitivity(LogisticGrowthSystem(), bounds, output='population.n', method='oat', until=20)
        >>> result.ranking[0]
        'growth.K'
    """
    if method not in METHODS:
        error_msg = f'Unknown sensitivity method {method!r}. Use one of {", ".join(METHODS)}.'
        raise AnalysisError(error_msg)
    if not isinstance(output, str):
        error_msg = f'output = {output!r} must be the name of a state or an observable of the system.'
        raise AnalysisError(error_msg)
    size = n if n is not None else SIZES[method]
    rows = _get_sensitivity_rows(bounds, method, size, random_state)
    result = _run_experiment(
        system,
        rows,
        [ScenarioSpec()],
        1,
        analyses=None,
        until=until,
        dt=dt,
        solver=solver,
        outputs=[output],
        warmup=0.0,
        observables=observables,
        random_state=random_state,
        executor=executor,
    )
    names = list(bounds)
    inputs = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    outputs = np.asarray(result.column(output), dtype=float)
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
