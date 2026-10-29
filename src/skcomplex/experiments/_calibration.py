"""Fit model parameters to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, minimize

from ..engine.runtime import Simulation
from ..errors import ValidationError
from ..types import Array
from ._base import BaseExperiment, Source

LOSSES = ('squares', 'likelihood')


class CalibrationExperiment(BaseExperiment):
    """Fit of parameters, given by dotted path, so that a model reproduces observed time series, with SciPy.

    Every evaluation runs the model to the last data time and interpolates its states or observables at the data
    times. Stochastic models average `replicates` runs with fixed streams, so the loss is smooth in the parameters.
    Least squares use `scipy.optimize.least_squares`, and the Gaussian likelihood with an unknown variance uses
    `scipy.optimize.minimize`.

    Args:
        source:
            A factory of systems, or a system, such as one that `skcomplex.models.create` returns.

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
            The time between frames, or `None` for the default of the simulation.

        random_state:
            The seed of the runs.

        arguments:
            The factory arguments of the model, or `None`.

    Attributes:
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
        >>> from skcomplex.experiments import CalibrationExperiment
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> times = np.arange(0, 21.0)
        >>> values = 100 / (1 + 9 * np.exp(-0.5 * times))
        >>> bounds = {'growth.r': (0.1, 2.0), 'growth.K': (50, 200)}
        >>> calibration = CalibrationExperiment(logistic_growth, {'population.n': (times, values)}, bounds)
        >>> estimates = calibration.check().prepare().run().estimates_
        >>> round(estimates['growth.r'], 2), round(estimates['growth.K'])
        (0.5, 100)
    """

    def __init__(
        self,
        source: Source,
        data: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
        bounds: Mapping[str, tuple[float, float]],
        *,
        initial: Mapping[str, float] | None = None,
        loss: str = 'squares',
        replicates: int = 1,
        dt: float | None = None,
        random_state: int | None = None,
        arguments: Mapping[str, object] | None = None,
    ) -> None:
        self.source = source
        self.data = data
        self.bounds = bounds
        self.initial = initial
        self.loss = loss
        self.replicates = replicates
        self.dt = dt
        self.random_state = random_state
        self.arguments = arguments

    def _check(self) -> None:
        """Raise an error for a model name as the source, an unknown loss, no data, or bad replicates."""
        self._check_source(self.source)
        if self.loss not in LOSSES:
            error_msg = f'Unknown loss {self.loss!r}. Use one of {", ".join(LOSSES)}.'
            raise ValidationError(error_msg)
        if not self.data:
            error_msg = 'Calibration needs at least one observed series.'
            raise ValidationError(error_msg)
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int) or self.replicates < 1:
            error_msg = f'replicates = {self.replicates!r} must be a positive integer.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Compile the model, read the data, and set the random streams."""
        factory = self._check_source(self.source)
        self.problem_ = self._compile(factory, self.arguments or {})
        self.series_ = {
            path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
            for path, (times, values) in self.data.items()
        }
        self.seeds_ = np.random.SeedSequence(self.random_state).spawn(self.replicates)

    def _simulate(self, values: Sequence[float]) -> dict[str, Array]:
        """Return the simulated series at the data times, averaged over the replicates, for parameter values."""
        candidate = self._with_values(self.problem_, dict(zip(self.bounds, map(float, values), strict=True)))
        until = max(float(times.max()) for times, _ in self.series_.values())
        runs = []
        for seed in self.seeds_:
            simulation = Simulation(candidate, dt=self.dt, random_state=np.random.default_rng(seed)).check()
            trajectory = simulation.run(until).trajectory_
            runs.append(
                {
                    path: np.interp(times, trajectory.times, np.asarray(trajectory.values(path), dtype=float))
                    for path, (times, _) in self.series_.items()
                },
            )
        return {path: np.mean([run[path] for run in runs], axis=0) for path in self.series_}

    def _residuals(self, values: Sequence[float]) -> Array:
        """Return the simulated minus the observed values of every series, concatenated."""
        simulated = self._simulate(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in self.series_.items()])

    def _run(self, inputs: None = None) -> Self:
        """Fit the parameters and set the estimates, their covariance, and the fit statistics."""
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
        return self
