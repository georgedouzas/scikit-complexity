"""Take sections of a system: Poincaré sections, stroboscopic samples, and orbit diagrams over a parameter."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Self

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..core.types import Vector
from ..engine.compile import compiled_problem
from ..engine.problems import CompiledEvent, EventRecord, Form, Problem
from ..engine.runtime import Simulation
from ..engine.solvers import Solver
from ..experiments import Experiment, GridDesign
from ._base import BaseAnalysis, Source
from ._bifurcation import _ensemble
from ._kinds import DataKind
from ._utils import check_kind, system_of

type Array = npt.NDArray[np.float64]
KINDS = ('poincare', 'stroboscopic', 'orbit-diagram')
SETTINGS = {
    'poincare': {'variables', 'state', 'value', 'direction', 'until', 'transient', 'solver', 'random_state'},
    'stroboscopic': {'variables', 'period', 'n_points', 'transient', 'solver', 'random_state'},
    'orbit-diagram': {
        'parameter',
        'parameter_values',
        'observable',
        'transient',
        'keep',
        'dt',
        'decimals',
        'solver',
        'random_state',
        'executor',
        'vectorized',
    },
}
"""Settings every kind reads, besides `kind`."""
DIRECTIONS = (-1, 0, 1)
SECTION = '__section'


class SectionAnalysis(BaseAnalysis):
    """Section of a system: the states where a flow crosses a value, at multiples of a period, or over a parameter.

    The kinds run the system:

    - `'poincare'`: the states where the scalar state `state` crosses `value` in a direction, located like an event.
    - `'stroboscopic'`: the states of a periodically forced flow at multiples of the forcing period.
    - `'orbit-diagram'`: the values an observable visits after a transient, for every value of a parameter, run as
      an experiment over a grid of the parameter or as one simulation of the system replicated over its values.

    Args:
        kind:
            The section to take, one of the kinds above.

        variables:
            The dotted names of the states whose values to read at the points of a section, or `None` for every state.

        state:
            The dotted name of the scalar state that defines a Poincaré section, such as `oscillator.x`.

        value:
            The value of the state on the Poincaré section.

        direction:
            `1` for crossings upward, `-1` for crossings downward, and `0` for both.

        until:
            The time a Poincaré section runs until.

        transient:
            The time before which crossings are discarded, the number of periods discarded before the first
            stroboscopic point, or the simulated time discarded before an orbit diagram keeps values. `None` gives 0
            and 100 periods, and an orbit diagram needs it.

        period:
            The period of the forcing, such as `2 * pi / omega`.

        n_points:
            The number of points of a stroboscopic section.

        parameter:
            The dotted path or factory argument an orbit diagram sweeps.

        parameter_values:
            The values of the parameter.

        observable:
            The state or observable path whose values an orbit diagram keeps.

        keep:
            The number of frames an orbit diagram keeps after the transient.

        dt:
            The time between frames of an orbit diagram, or `None` for the model's default, 1 for maps.

        decimals:
            The decimals the kept values are rounded to before counting the period of every orbit.

        solver:
            The solver, as in `Simulation`.

        random_state:
            The seed of stochastic solvers.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor of the runs of an orbit diagram.

        vectorized:
            Whether to run an orbit diagram as one simulation of the system replicated over the values, which needs a
            dotted parameter path and a system of single elements.

    Attributes:
        problem_:
            The compiled problem of a Poincaré or stroboscopic section.

        source_:
            The registered model name, factory, or system of an orbit diagram.

        times_:
            The time of every point of a section.

        states_:
            One state vector per point of a section.

        values_:
            A mapping from the name of every read state to its values at the points of a section, one row per point
            for a vector state.

        points_:
            The kept values of the observable for every value of the parameter of an orbit diagram.

        periods_:
            The number of distinct kept values for every value of the parameter, the period of periodic orbits.

    Examples:
        >>> from skcomplex.analysis import SectionAnalysis
        >>> from skcomplex.models.physics.mechanics import harmonic_oscillator
        >>> section = SectionAnalysis(state='mass.x', until=20.0).check().prepare(harmonic_oscillator()).run()
        >>> len(section.times_) > 2
        True
        >>> diagram = SectionAnalysis(
        ...     'orbit-diagram', parameter='map.r', parameter_values=[2.9, 3.2], observable='map.x', transient=1000,
        ...     keep=20, dt=1,
        ... )
        >>> diagram.check().prepare('logistic_map').run().periods_.tolist()
        [1, 2]
    """

    def __init__(
        self,
        kind: str = 'poincare',
        variables: Sequence[str] | None = None,
        *,
        state: str | None = None,
        value: float = 0.0,
        direction: int = 1,
        until: float = 1000.0,
        transient: float | None = None,
        period: float | None = None,
        n_points: int = 1000,
        parameter: str | None = None,
        parameter_values: Sequence[float] | None = None,
        observable: str | None = None,
        keep: int | None = None,
        dt: float | None = None,
        decimals: int = 6,
        solver: Solver | None = None,
        random_state: int | None = None,
        executor: str | Executor = 'serial',
        vectorized: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.state = state
        self.value = value
        self.direction = direction
        self.until = until
        self.transient = transient
        self.period = period
        self.n_points = n_points
        self.parameter = parameter
        self.parameter_values = parameter_values
        self.observable = observable
        self.keep = keep
        self.dt = dt
        self.decimals = decimals
        self.solver = solver
        self.random_state = random_state
        self.executor = executor
        self.vectorized = vectorized

    def _check(self) -> Self:
        """Validate the kind, that the kind reads every changed setting, and that its needed settings are given."""
        check_kind(self, KINDS, SETTINGS)
        needed = {
            'poincare': {'state': self.state},
            'stroboscopic': {'period': self.period},
            'orbit-diagram': {
                'parameter': self.parameter,
                'parameter_values': self.parameter_values,
                'observable': self.observable,
                'transient': self.transient,
                'keep': self.keep,
            },
        }[self.kind]
        missing = [name for name, value in needed.items() if value is None]
        if missing:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs {", ".join(missing)}. Give them.'
            raise AnalysisError(error_msg)
        if self.direction not in DIRECTIONS:
            error_msg = f'{type(self).__name__} has direction={self.direction!r}. Use one of -1, 0, or 1.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile a flow for a section, or keep the source of an orbit diagram."""
        if self.kind == 'orbit-diagram':
            self.source_ = source
            return self
        problem = compiled_problem(system_of(source), 'A section')
        if problem.form not in {Form.ODE, Form.SDE}:
            error_msg = f'Sections are taken of flows in continuous time, but the system is a {problem.form.value}.'
            raise AnalysisError(error_msg)
        if self.kind == 'poincare' and (
            not problem.states.has(self.state) or problem.states.entry(self.state).size != 1
        ):
            error_msg = (
                f'{self.state!r} is not a scalar state. Available states are: {", ".join(problem.states.names)}.'
            )
            raise AnalysisError(error_msg)
        self.problem_ = problem
        return self

    def _crossings(self, problem: Problem) -> tuple[Array, Array]:
        """Return the times and the states where the state crosses the value of the Poincaré section."""
        index = problem.states.entry(self.state).start
        value = self.value

        def guard(t: float, x: Vector, p: Vector) -> float:
            return float(x[index] - value)

        transient = self.transient if self.transient is not None else 0.0
        event = CompiledEvent(SECTION, guard, None, int(self.direction), False)
        sectioned = dataclasses.replace(problem, events=(*problem.events, event))
        trajectory = (
            Simulation(sectioned, solver=self.solver, dt=self.until / 100, random_state=self.random_state)
            .run(self.until)
            .trajectory_
        )
        crossings = [
            record
            for record in trajectory.events
            if isinstance(record, EventRecord) and record.label == SECTION and record.time >= transient
        ]
        states = np.zeros((len(crossings), problem.states.size))
        for row, record in enumerate(crossings):
            states[row] = record.before
        return np.asarray([record.time for record in crossings], dtype=np.float64), states

    def _samples(self, problem: Problem) -> tuple[Array, Array]:
        """Return the times and the states at multiples of the forcing period."""
        period = float(self.period)  # type: ignore[arg-type]
        transient = int(self.transient if self.transient is not None else 100)
        until = period * (transient + self.n_points - 1)
        simulation = Simulation(problem, solver=self.solver, dt=period, random_state=self.random_state)
        trajectory = simulation.run(until).trajectory_
        return trajectory.times[transient:], trajectory.states[transient:]

    def _orbits(self) -> list[Array]:
        """Return the kept values of the observable for every value of the parameter."""
        step = self.dt if self.dt is not None else 1.0
        transient = float(self.transient)  # type: ignore[arg-type]
        keep = int(self.keep)  # type: ignore[arg-type]
        values = list(self.parameter_values)  # type: ignore[arg-type]
        if self.vectorized:
            settings = {'dt': step, 'random_state': self.random_state}
            trajectory, _ = _ensemble(self.source_, {self.parameter: values}, transient + (keep - 1) * step, settings)
            series = np.asarray(trajectory.values(self.observable))[np.asarray(trajectory.times) >= transient - 1e-9]
            return [series[:, index] for index in range(len(values))]
        experiment = Experiment(
            self.source_,
            GridDesign(**{self.parameter: values}),
            until=transient + (keep - 1) * step,
            dt=self.dt,
            solver=self.solver,
            outputs=[self.observable],
            warmup=transient,
            random_state=self.random_state,
            executor=self.executor,
        )
        return [run.series[self.observable] for run in experiment.run().runs]

    def _run(self) -> Self:
        """Run the system and take the section, or the orbit diagram."""
        if self.kind == 'orbit-diagram':
            self.points_ = self._orbits()
            self.periods_ = np.asarray(
                [np.unique(np.round(points, self.decimals)).size for points in self.points_],
                dtype=np.int64,
            )
            return self
        problem = self.problem_
        self.times_, self.states_ = self._crossings(problem) if self.kind == 'poincare' else self._samples(problem)
        names = list(problem.states.names) if self.variables is None else self.variables
        self.values_ = {}
        for name in names:
            values = np.asarray([problem.states.read(state, name) for state in self.states_], dtype=float)
            size = problem.states.entry(name).size
            self.values_[name] = values.reshape(len(self.times_), -1) if size > 1 else values.reshape(-1)
        return self
