"""Fit model parameters to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, minimize

from ..core import ValidationError
from ..runtime import Simulation
from ._runner import Source, _problem, resolve_source, with_values

LOSSES = ('squares', 'likelihood')


@dataclass(frozen=True)
class CalibrationResult:
    """Parameters fitted to data, with their uncertainty and the quality of the fit.

    Args:
        estimates:
            The fitted value of every parameter.

        covariance:
            The covariance of the estimates from the Jacobian of the residuals, or `None` when it is singular.

        r2:
            The coefficient of determination over all data points.

        rmse:
            The root mean squared error.

        mae:
            The mean absolute error.

        mape:
            The mean absolute percentage error, over data points that are not zero.

        simulated:
            The simulated values at the data times, by path.

        success:
            Whether the optimiser converged.
    """

    estimates: dict[str, float]
    covariance: npt.NDArray[np.float64] | None
    r2: float
    rmse: float
    mae: float
    mape: float
    simulated: dict[str, npt.NDArray[np.float64]]
    success: bool


def calibrate(
    source: Source,
    data: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    bounds: Mapping[str, tuple[float, float]],
    *,
    initial: Mapping[str, float] | None = None,
    loss: str = 'squares',
    replicates: int = 1,
    dt: float | None = None,
    random_state: int | None = None,
    **arguments: Any,  # noqa: ANN401
) -> CalibrationResult:
    """Fit parameters, given by dotted path, so that a model reproduces observed time series.

    Every evaluation runs the model to the last data time and interpolates its states or observables at the data
    times. Stochastic models average `replicates` runs with fixed streams, so the loss is smooth in the parameters.

    Args:
        source:
            A registered model name, a factory, or a system.

        data:
            A mapping from state or observable path to its observed `(times, values)`.

        bounds:
            A mapping from dotted parameter or initial-state path to its `(low, high)` bounds.

        initial:
            The starting values of the parameters, or `None` for the midpoints of their bounds.

        loss:
            `'squares'` for least squares or `'likelihood'` for the Gaussian likelihood with an unknown variance.

        replicates:
            The number of runs averaged per evaluation.

        dt:
            The time between frames, or `None` for the model's default.

        random_state:
            The seed of the runs.

        **arguments:
            Factory arguments of the model.

    Returns:
        result:
            The estimates, their covariance, the fit statistics, and the simulated values at the data times.

    Raises:
        ValidationError:
            The loss is unknown or there are no data.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.experiments import calibrate
        >>> from skcomplex.models.ecology import logistic_growth
        >>> times = np.arange(0, 21.0)
        >>> values = 100 / (1 + 9 * np.exp(-0.5 * times))
        >>> result = calibrate('logistic_growth', {'population.n': (times, values)},
        ...                    {'growth.r': (0.1, 2.0), 'growth.K': (50, 200)})
        >>> round(result.estimates['growth.r'], 2), round(result.estimates['growth.K'])
        (0.5, 100)
    """
    if loss not in LOSSES:
        error_msg = f'Unknown loss {loss!r}. Use one of {", ".join(LOSSES)}.'
        raise ValidationError(error_msg)
    if not data:
        error_msg = 'Calibration needs at least one observed series.'
        raise ValidationError(error_msg)
    factory, defaults = resolve_source(source)
    problem = _problem(factory, arguments, None)
    names = list(bounds)
    series = {
        path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for path, (times, values) in data.items()
    }
    until = max(float(times.max()) for times, _ in series.values())
    settings: dict[str, Any] = {'dt': dt if dt is not None else defaults.get('dt')}
    settings = {key: value for key, value in settings.items() if value is not None}
    if 'solver' in defaults:
        settings['solver'] = defaults['solver']
    seeds = np.random.SeedSequence(random_state).spawn(replicates)

    def simulate(values: Sequence[float]) -> dict[str, npt.NDArray[np.float64]]:
        candidate = with_values(problem, dict(zip(names, map(float, values), strict=True)))
        runs = []
        for seed in seeds:
            trajectory = (
                Simulation(candidate, random_state=np.random.default_rng(seed), **settings).run(until).trajectory_
            )
            runs.append(
                {
                    path: np.interp(times, trajectory.times, np.asarray(trajectory[path], dtype=float))
                    for path, (times, _) in series.items()
                },
            )
        return {path: np.mean([run[path] for run in runs], axis=0) for path in series}

    def residuals(values: Sequence[float]) -> npt.NDArray[np.float64]:
        simulated = simulate(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in series.items()])

    start = np.asarray([(initial or {}).get(name, sum(bounds[name]) / 2) for name in names], dtype=float)
    low = np.asarray([bounds[name][0] for name in names], dtype=float)
    high = np.asarray([bounds[name][1] for name in names], dtype=float)
    if loss == 'squares':
        fit = least_squares(residuals, start, bounds=(low, high), x_scale='jac')
        estimate, success, jacobian = fit.x, bool(fit.success), fit.jac
    else:
        count = sum(observed.size for _, observed in series.values())
        fit = minimize(
            lambda values: 0.5 * count * np.log(np.sum(residuals(values) ** 2)),
            start,
            bounds=list(zip(low, high, strict=True)),
        )
        estimate, success = fit.x, bool(fit.success)
        jacobian = least_squares(residuals, estimate, bounds=(low, high), max_nfev=1).jac
    final = residuals(estimate)
    observed = np.concatenate([values for _, values in series.values()])
    dof = max(observed.size - len(names), 1)
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(final**2)) / dof
    except np.linalg.LinAlgError:
        covariance = None
    nonzero = observed != 0
    return CalibrationResult(
        estimates=dict(zip(names, map(float, estimate), strict=True)),
        covariance=covariance,
        r2=float(1 - np.sum(final**2) / np.sum((observed - observed.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(final**2))),
        mae=float(np.mean(np.abs(final))),
        mape=float(100 * np.mean(np.abs(final[nonzero] / observed[nonzero]))) if nonzero.any() else float('nan'),
        simulated=simulate(estimate),
        success=success,
    )
