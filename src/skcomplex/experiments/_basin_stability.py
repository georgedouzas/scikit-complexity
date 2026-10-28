"""Estimate the basin stability of the attractors of a model over random initial states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from typing import Self

import numpy as np

from ..core.errors import ValidationError
from ..engine.compile import compile_system
from ..engine.problems import Problem
from ..engine.runtime import Trajectory
from ._base import BaseExperiment, Source
from ._experiment import Experiment
from ._random import RandomDesign


class BasinStabilityExperiment(BaseExperiment):
    """Fractions of random initial states, drawn uniformly within bounds, that end in each attractor.

    Every initial state is run to `until` and its trajectory is labelled by the classifier. With `vectorized`, all
    initial states run as the members of one simulation of the replicated system, and the classifier receives every
    member as a trajectory of the original system.

    Args:
        source:
            A factory of systems, or a system, such as one that `skcomplex.models.create` returns.

        bounds:
            A mapping from dotted state path to the `(low, high)` bounds of its initial value.

        classifier:
            A function of a trajectory that returns the label of the attractor it ends in.

        n:
            The number of initial states.

        until:
            The simulated time of every run, or `None` for the default of the simulation.

        dt:
            The time between frames, or `None` for the default of the simulation.

        random_state:
            The seed of the initial states and the runs.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the separate runs.

        vectorized:
            Whether to run every initial state as a member of one simulation of the replicated system.

    Attributes:
        rows_:
            The initial states, one mapping from dotted path to value per run, set by `prepare`.

        labels_:
            The attractor label of every initial state.

        fractions_:
            The fraction of initial states per attractor label.

        errors_:
            The standard error of every fraction.

    Examples:
        >>> from skcomplex.experiments import BasinStabilityExperiment
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> experiment = BasinStabilityExperiment(
        ...     logistic_growth, {'population.n': (0.1, 10)}, lambda trajectory: 'capacity', n=5, until=5
        ... )
        >>> experiment.check().prepare().run().fractions_
        {'capacity': 1.0}
    """

    def __init__(
        self,
        source: Source,
        bounds: Mapping[str, tuple[float, float]],
        classifier: Callable[[Trajectory], object],
        *,
        n: int = 500,
        until: float | None = None,
        dt: float | None = None,
        random_state: int | None = None,
        executor: str | Executor = 'serial',
        vectorized: bool = False,
    ) -> None:
        self.source = source
        self.bounds = bounds
        self.classifier = classifier
        self.n = n
        self.until = until
        self.dt = dt
        self.random_state = random_state
        self.executor = executor
        self.vectorized = vectorized

    def _check(self) -> None:
        """Raise an error for a model name as the source, no bounds, or a bad number of initial states."""
        self._check_source(self.source)
        if not self.bounds:
            error_msg = 'Basin stability needs the bounds of at least one initial state.'
            raise ValidationError(error_msg)
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            error_msg = f'n = {self.n!r} must be a positive integer.'
            raise ValidationError(error_msg)

    def _prepare(self, context: object = None) -> None:
        """Draw the initial states within the bounds."""
        self.rows_ = RandomDesign(n=self.n, random_state=self.random_state).check().check().prepare(self.bounds).run()

    def _run(self, inputs: object = None) -> Self:
        """Run every initial state, label its trajectory, and set the fraction and error of every attractor."""
        self.labels_ = self._label_members() if self.vectorized else self._label_runs()
        counts = Counter(self.labels_)
        self.fractions_ = {label: count / self.n for label, count in counts.items()}
        self.errors_ = {
            label: float(np.sqrt(fraction * (1 - fraction) / self.n)) for label, fraction in self.fractions_.items()
        }
        return self

    def _label_runs(self) -> list[object]:
        """Return the label of the separate run of every initial state."""
        classifier = self.classifier

        def attractor(trajectory: Trajectory) -> object:
            return classifier(trajectory)

        experiment = Experiment(
            self.source,
            RandomDesign(n=self.n, random_state=self.random_state),
            self.bounds,
            until=self.until,
            dt=self.dt,
            outputs=[attractor],
            random_state=self.random_state,
            executor=self.executor,
        )
        return [row['attractor'] for row in experiment.check().prepare().run().result_.rows]

    def _label_members(self) -> list[object]:
        """Return the label of every member of one simulation of the system replicated over the initial states."""
        values = {path: [row[path] for row in self.rows_] for path in self.bounds}
        trajectory, system = self._run_replicated(self.source, values, self.until, self.dt, self.random_state)
        problem = compile_system(system)
        if not isinstance(problem, Problem):
            error_msg = f'A vectorised basin reads the members of a general problem, not of a {type(problem).__name__}.'
            raise ValidationError(error_msg)
        return [self.classifier(trajectory.member(index, problem)) for index in range(self.n)]
