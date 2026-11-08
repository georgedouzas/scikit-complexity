"""Fit the model parameters of a system to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, minimize

from ..components.system import BaseSystem
from ..engine.runtime import simulate
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..types import Array
from ._runs import _build, _check_count, _check_source
from .types import Source

LOSSES = ('squares', 'likelihood')


@dataclass(frozen=True)
class CalibrationResult:
    """Fit of the model parameters of a system to observed time series.

    Args:
        system:
            The system with the fitted model parameters.

        estimates:
            The fitted value of every parameter.

        r2:
            The coefficient of determination over all data points.

        rmse:
            The root mean squared error.

        mae:
            The mean absolute error.

        mape:
            The mean absolute percentage error, over data points that are not zero.

        covariance:
            The covariance of the estimates from the Jacobian of the residuals, or `None` when it is singular.

        simulated:
            The simulated values at the data times, by path.

        success:
            Whether the optimiser converged.
    """

    system: BaseSystem
    estimates: dict[str, float]
    r2: float
    rmse: float
    mae: float
    mape: float
    covariance: Array | None
    simulated: dict[str, Array]
    success: bool


def calibrate(
    system: Source,
    data: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    bounds: Mapping[str, tuple[float, float]],
    *,
    initial: Mapping[str, float] | None = None,
    loss: str = 'squares',
    replicates: int = 1,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
) -> CalibrationResult:
    """Return the fit of the model parameters of a system to observed time series, with SciPy.

    The fitted system reproduces the observed time series. It is a copy of the system with `set_params` applied, or
    what the factory builds from the parameters. Every evaluation runs the system through `simulate` to the last data
    time and interpolates its states or observables at the data times. Stochastic systems average `replicates` runs
    with fixed streams, so the loss is smooth in the parameters. Least squares use `scipy.optimize.least_squares`, and
    the Gaussian likelihood with an unknown variance uses `scipy.optimize.minimize`.

    Args:
        system:
            The system, whose copies take the model parameters through `set_params`, or a factory of systems, such as
            a system class, which takes them as keyword arguments.

        data:
            A mapping from state or observable path to its observed `(times, values)`.

        bounds:
            A mapping from model parameter of the system to its `(low, high)` bounds.

        initial:
            The starting values of the parameters, or `None` for the midpoints of their bounds.

        loss:
            `'squares'` for least squares or `'likelihood'` for the Gaussian likelihood with an unknown variance.

        replicates:
            The number of runs averaged per evaluation.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the runs.

    Returns:
        result:
            The fitted system, the estimates, and the goodness of fit.

    Raises:
        ValidationError:
            The system is a system name, the loss is unknown, there is no data, the replicates are bad, or a bound has
            a dotted path.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.experiments import calibrate
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> times = np.arange(0, 21.0)
        >>> values = 100 / (1 + 9 * np.exp(-0.5 * times))
        >>> bounds = {'r': (0.1, 2.0), 'K': (50, 200)}
        >>> fitted = calibrate(LogisticGrowthSystem(), {'population.n': (times, values)}, bounds).system
        >>> round(fitted.r, 2), round(fitted.K)
        (0.5, 100)
    """
    _check_source(system)
    if loss not in LOSSES:
        error_msg = f'Unknown loss {loss!r}. Use one of {", ".join(LOSSES)}.'
        raise ValidationError(error_msg)
    if not data:
        error_msg = 'Calibration needs at least one observed series.'
        raise ValidationError(error_msg)
    _check_count('replicates', replicates)
    dotted = sorted(name for name in bounds if '.' in name)
    if dotted:
        error_msg = (
            f'Calibration fits model parameters of the system, which `set_params` takes, not the dotted paths '
            f'{", ".join(dotted)}.'
        )
        raise ValidationError(error_msg)
    series = {
        path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for path, (times, values) in data.items()
    }
    seeds = np.random.SeedSequence(random_state).spawn(replicates)
    names = list(bounds)

    def fitted_system(values: Sequence[float]) -> BaseSystem:
        return _build(system, dict(zip(names, map(float, values), strict=True)))

    def simulated_at(values: Sequence[float]) -> dict[str, Array]:
        candidate = fitted_system(values)
        until = max(float(times.max()) for times, _ in series.values())
        runs = []
        for seed in seeds:
            trajectory = simulate(
                candidate,
                until=until,
                dt=dt,
                solver=solver,
                random_state=np.random.default_rng(seed),
            )
            runs.append(
                {
                    path: np.interp(times, trajectory.times, np.asarray(trajectory.values(path), dtype=float))
                    for path, (times, _) in series.items()
                },
            )
        return {path: np.mean([run[path] for run in runs], axis=0) for path in series}

    def residuals(values: Sequence[float]) -> Array:
        simulated = simulated_at(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in series.items()])

    start = np.asarray([(initial or {}).get(name, sum(bounds[name]) / 2) for name in names], dtype=float)
    low = np.asarray([bounds[name][0] for name in names], dtype=float)
    high = np.asarray([bounds[name][1] for name in names], dtype=float)
    if loss == 'squares':
        fit = least_squares(residuals, start, bounds=(low, high), x_scale='jac')
        estimate, jacobian = fit.x, fit.jac
    else:
        count = sum(observed.size for _, observed in series.values())
        fit = minimize(
            lambda values: 0.5 * count * np.log(np.sum(residuals(values) ** 2)),
            start,
            bounds=list(zip(low, high, strict=True)),
        )
        estimate = fit.x
        jacobian = least_squares(residuals, estimate, bounds=(low, high), max_nfev=1).jac
    final = residuals(estimate)
    observed = np.concatenate([values for _, values in series.values()])
    dof = max(observed.size - len(names), 1)
    covariance: Array | None
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(final**2)) / dof
    except np.linalg.LinAlgError:
        covariance = None
    nonzero = observed != 0
    return CalibrationResult(
        system=fitted_system(estimate),
        estimates=dict(zip(names, map(float, estimate), strict=True)),
        r2=float(1 - np.sum(final**2) / np.sum((observed - observed.mean()) ** 2)),
        rmse=float(np.sqrt(np.mean(final**2))),
        mae=float(np.mean(np.abs(final))),
        mape=float(100 * np.mean(np.abs(final[nonzero] / observed[nonzero]))) if nonzero.any() else float('nan'),
        covariance=covariance,
        simulated=simulated_at(estimate),
        success=bool(fit.success),
    )
