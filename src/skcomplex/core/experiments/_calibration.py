"""Fit the model parameters of a system to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, minimize

from ..base import BaseConfig
from ..components.system import BaseSystem
from ..engine.runtime import simulate
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..types import Array
from ._runs import _build, _check_count, _check_source
from .types import Source

LOSSES = ('squares', 'likelihood')


class Calibration(BaseConfig[None, None]):
    """Fit of the model parameters of a system to observed time series, with SciPy.

    The fitted system reproduces the observed time series. Calibration is an estimator in the role of `GridSearchCV` of
    scikit-learn: `run` fits the model parameters and sets `system_`, the system with the fitted parameters, a copy of
    the system with `set_params` applied or what the factory builds from them. Every evaluation runs the system through
    `simulate` to the last data time and interpolates its states or observables at the data times. Stochastic systems
    average `replicates` runs with fixed streams, so the loss is smooth in the parameters. Least squares use
    `scipy.optimize.least_squares`, and the Gaussian likelihood with an unknown variance uses `scipy.optimize.minimize`.

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

    Attributes:
        system_:
            The system with the fitted model parameters.

        estimates_:
            The fitted value of every parameter.

        covariance_:
            The covariance of the estimates from the Jacobian of the residuals, or `None` when it is singular.

        r2_:
            The coefficient of determination over all data points.

        rmse_:
            The root mean squared error.

        mae_:
            The mean absolute error.

        mape_:
            The mean absolute percentage error, over data points that are not zero.

        simulated_:
            The simulated values at the data times, by path.

        success_:
            Whether the optimiser converged.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.experiments import Calibration
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> times = np.arange(0, 21.0)
        >>> values = 100 / (1 + 9 * np.exp(-0.5 * times))
        >>> bounds = {'r': (0.1, 2.0), 'K': (50, 200)}
        >>> calibration = Calibration(LogisticGrowthSystem(), {'population.n': (times, values)}, bounds)
        >>> fitted = calibration.prepare().run().system_
        >>> round(fitted.r, 2), round(fitted.K)
        (0.5, 100)
    """

    def __init__(
        self,
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
    ) -> None:
        self.system = system
        self.data = data
        self.bounds = bounds
        self.initial = initial
        self.loss = loss
        self.replicates = replicates
        self.dt = dt
        self.solver = solver
        self.random_state = random_state

    def _prepare(self, context: None = None) -> None:
        """Read the data and set the random streams.

        It first raises an error for an invalid system, loss, data, replicates, or bounds. The system is a system name,
        the loss is unknown, there is no data, the replicates are bad, or a bound has a dotted path.
        """
        _check_source(self.system)
        if self.loss not in LOSSES:
            error_msg = f'Unknown loss {self.loss!r}. Use one of {", ".join(LOSSES)}.'
            raise ValidationError(error_msg)
        if not self.data:
            error_msg = 'Calibration needs at least one observed series.'
            raise ValidationError(error_msg)
        _check_count('replicates', self.replicates)
        dotted = sorted(name for name in self.bounds if '.' in name)
        if dotted:
            error_msg = (
                f'Calibration fits model parameters of the system, which `set_params` takes, not the dotted paths '
                f'{", ".join(dotted)}.'
            )
            raise ValidationError(error_msg)
        self.series_ = {
            path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
            for path, (times, values) in self.data.items()
        }
        self.seeds_ = np.random.SeedSequence(self.random_state).spawn(self.replicates)

    def _simulate(self, values: Sequence[float]) -> dict[str, Array]:
        """Return the simulated series at the data times for parameter values.

        They are averaged over the replicates.
        """
        candidate = self._fitted_system(values)
        until = max(float(times.max()) for times, _ in self.series_.values())
        runs = []
        for seed in self.seeds_:
            trajectory = simulate(
                candidate,
                until=until,
                dt=self.dt,
                solver=self.solver,
                random_state=np.random.default_rng(seed),
            )
            runs.append(
                {
                    path: np.interp(times, trajectory.times, np.asarray(trajectory.values(path), dtype=float))
                    for path, (times, _) in self.series_.items()
                },
            )
        return {path: np.mean([run[path] for run in runs], axis=0) for path in self.series_}

    def _fitted_system(self, values: Sequence[float]) -> BaseSystem:
        """Return the system with model parameter values, in the order of the bounds."""
        return _build(self.system, dict(zip(self.bounds, map(float, values), strict=True)))

    def _residuals(self, values: Sequence[float]) -> Array:
        """Return the simulated minus the observed values of every series, concatenated."""
        simulated = self._simulate(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in self.series_.items()])

    def _run(self, inputs: None = None) -> None:
        """Fit the parameters and set the estimates and their covariance.

        It also sets the fit statistics.
        """
        names = list(self.bounds)
        start = np.asarray([(self.initial or {}).get(name, sum(self.bounds[name]) / 2) for name in names], dtype=float)
        low = np.asarray([self.bounds[name][0] for name in names], dtype=float)
        high = np.asarray([self.bounds[name][1] for name in names], dtype=float)
        if self.loss == 'squares':
            fit = least_squares(self._residuals, start, bounds=(low, high), x_scale='jac')
            estimate, jacobian = fit.x, fit.jac
        else:
            count = sum(observed.size for _, observed in self.series_.values())
            fit = minimize(
                lambda values: 0.5 * count * np.log(np.sum(self._residuals(values) ** 2)),
                start,
                bounds=list(zip(low, high, strict=True)),
            )
            estimate = fit.x
            jacobian = least_squares(self._residuals, estimate, bounds=(low, high), max_nfev=1).jac
        final = self._residuals(estimate)
        observed = np.concatenate([values for _, values in self.series_.values()])
        dof = max(observed.size - len(names), 1)
        self.covariance_: Array | None
        try:
            self.covariance_ = np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(final**2)) / dof
        except np.linalg.LinAlgError:
            self.covariance_ = None
        nonzero = observed != 0
        self.estimates_ = dict(zip(names, map(float, estimate), strict=True))
        self.r2_ = float(1 - np.sum(final**2) / np.sum((observed - observed.mean()) ** 2))
        self.rmse_ = float(np.sqrt(np.mean(final**2)))
        self.mae_ = float(np.mean(np.abs(final)))
        self.mape_ = float(100 * np.mean(np.abs(final[nonzero] / observed[nonzero]))) if nonzero.any() else np.nan
        self.simulated_ = self._simulate(estimate)
        self.success_ = bool(fit.success)
        self.system_ = self._fitted_system(estimate)
