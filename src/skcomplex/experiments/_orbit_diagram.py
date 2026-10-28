"""Build orbit diagrams: the values an observable visits after a transient, over a range of a parameter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import ValidationError
from ._base import BaseExperiment, Source
from ._experiment import Experiment
from ._grid import GridDesign

type Array = npt.NDArray[np.float64]


class OrbitDiagramExperiment(BaseExperiment):
    """Values an observable visits after a transient, for every value of a parameter.

    Every value of the parameter is run to `transient + (keep - 1) * dt`, and the frames from `transient` on are kept.
    With `vectorized`, all values run as the members of one simulation of the system replicated over them, which needs
    a dotted parameter path and a system of single elements.

    Args:
        source:
            A registered model name, a factory, or a system.

        parameter:
            The dotted path or factory argument to sweep.

        values:
            The values of the parameter.

        observable:
            The state or observable path whose values are kept.

        transient:
            The simulated time discarded before keeping values.

        keep:
            The number of frames kept after the transient.

        dt:
            The time between frames, or `None` for the model's default, 1 for maps.

        decimals:
            The decimals the kept values are rounded to before counting the period of every orbit.

        random_state:
            The seed of stochastic runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the separate runs.

        vectorized:
            Whether to run every value as a member of one simulation of the replicated system.

    Attributes:
        points_:
            The kept values of the observable for every value of the parameter.

        periods_:
            The number of distinct kept values for every value of the parameter, the period of periodic orbits.

    Examples:
        >>> from skcomplex.experiments import OrbitDiagramExperiment
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> diagram = OrbitDiagramExperiment('logistic_map', 'map.r', [2.9, 3.2], 'map.x', transient=1000, keep=20)
        >>> diagram.check().prepare().run().periods_.tolist()
        [1, 2]
    """

    def __init__(
        self,
        source: Source,
        parameter: str,
        values: Sequence[float],
        observable: str,
        *,
        transient: float,
        keep: int,
        dt: float | None = None,
        decimals: int = 6,
        random_state: int | None = None,
        executor: str | Executor = 'serial',
        vectorized: bool = False,
    ) -> None:
        self.source = source
        self.parameter = parameter
        self.values = values
        self.observable = observable
        self.transient = transient
        self.keep = keep
        self.dt = dt
        self.decimals = decimals
        self.random_state = random_state
        self.executor = executor
        self.vectorized = vectorized

    def _check(self) -> None:
        """Raise an error for no values or a number of kept frames that is not a positive integer."""
        if not len(self.values):
            error_msg = 'An orbit diagram needs at least one value of the parameter.'
            raise ValidationError(error_msg)
        if isinstance(self.keep, bool) or not isinstance(self.keep, int) or self.keep < 1:
            error_msg = f'keep = {self.keep!r} must be a positive integer.'
            raise ValidationError(error_msg)

    def _prepare(self) -> None:
        """Set the simulated time of every run."""
        self.until_ = self.transient + (self.keep - 1) * (self.dt if self.dt is not None else 1.0)

    def _run(self) -> Self:
        """Run every value of the parameter and keep the values of the observable after the transient."""
        self.points_ = self._replicated_points() if self.vectorized else self._separate_points()
        self.periods_ = np.asarray(
            [np.unique(np.round(points, self.decimals)).size for points in self.points_],
            dtype=np.int64,
        )
        return self

    def _separate_points(self) -> list[Array]:
        """Return the kept values of a separate run of every value of the parameter."""
        experiment = Experiment(
            self.source,
            GridDesign(),
            {self.parameter: list(self.values)},
            until=self.until_,
            dt=self.dt,
            outputs=[self.observable],
            warmup=self.transient,
            random_state=self.random_state,
            executor=self.executor,
        )
        result = experiment.check().prepare().run().result_
        return [run.series[self.observable] for run in result.runs]

    def _replicated_points(self) -> list[Array]:
        """Return the kept values of every member of one simulation of the system replicated over the values."""
        dt = self.dt if self.dt is not None else 1.0
        values = {self.parameter: list(self.values)}
        trajectory, _ = self._run_replicated(self.source, values, self.until_, dt, self.random_state)
        series = np.asarray(trajectory.values(self.observable))[np.asarray(trajectory.times) >= self.transient - 1e-9]
        return [series[:, index] for index in range(len(self.values))]
