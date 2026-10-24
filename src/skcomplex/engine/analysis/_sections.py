"""Take Poincaré sections of flows where a state crosses a value, and stroboscopic samples of forced systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError, Form
from ..compile import CompiledEvent, EventRecord, Problem, Vector, compile_system
from ..runtime import Simulation
from ..solvers import Stepper
from ..system import System

type Array = npt.NDArray[np.float64]

SECTION = '__section'


@dataclass(frozen=True)
class Section:
    """States of a system at the points of a section.

    Args:
        times:
            The time of every point.

        states:
            One state vector per point.

        problem:
            The compiled problem, which gives the names of the vector components.
    """

    times: Array
    states: Array
    problem: Problem

    def __len__(self) -> int:
        """Return the number of points."""
        return len(self.times)

    def __getitem__(self, name: str) -> Array:
        """Return the values of a state at every point by dotted name.

        Args:
            name:
                The dotted name of the state, such as `oscillator.x`.

        Returns:
            values:
                One value per point, or one row per point for a vector state.
        """
        values = np.asarray([self.problem.states.read(state, name) for state in self.states], dtype=float)
        return values.reshape(len(self), -1) if self.problem.states.entry(name).size > 1 else values.reshape(-1)


def _problem(system: System | Problem) -> Problem:
    """Return the compiled problem of a system after checking that it flows in continuous time."""
    problem = system if isinstance(system, Problem) else compile_system(system)
    if problem.form not in {Form.ODE, Form.SDE}:
        error_msg = f'Sections are taken of flows in continuous time, but the system is a {problem.form.value}.'
        raise AnalysisError(error_msg)
    return problem


def poincare_section(
    system: System | Problem,
    state: str,
    value: float = 0.0,
    *,
    direction: int = 1,
    until: float = 1000.0,
    transient: float = 0.0,
    solver: str | Stepper = 'auto',
    random_state: int | None = None,
) -> Section:
    """Return the states where a state crosses a value in a direction, located like an event.

    Args:
        system:
            The system, or a problem compiled from one.

        state:
            The dotted name of the scalar state that defines the section, such as `oscillator.x`.

        value:
            The value of the state on the section.

        direction:
            `1` for crossings upward, `-1` for crossings downward, and `0` for both.

        until:
            The time to run until.

        transient:
            The time before which crossings are discarded.

        solver:
            The solver, as in `Simulation`.

        random_state:
            The seed of stochastic solvers.

    Returns:
        section:
            The times and states of the crossings.

    Raises:
        AnalysisError:
            The system is not a flow, or the state is not a scalar state.

    Examples:
        >>> from skcomplex.core.analysis import poincare_section
        >>> from skcomplex.models.physics.mechanics import harmonic_oscillator
        >>> section = poincare_section(harmonic_oscillator(), 'mass.x', direction=1, until=20.0)
        >>> len(section) > 2
        True
    """
    problem = _problem(system)
    if state not in problem.states or problem.states.entry(state).size != 1:
        error_msg = f'{state!r} is not a scalar state. Available states are: {", ".join(problem.states.names)}.'
        raise AnalysisError(error_msg)
    index = problem.states.entry(state).start

    def guard(t: float, x: Vector, p: Vector) -> float:
        return float(x[index] - value)

    event = CompiledEvent(SECTION, guard, None, int(direction), False)
    sectioned = dataclasses.replace(problem, events=(*problem.events, event))
    trajectory = Simulation(sectioned, solver=solver, dt=until / 100, random_state=random_state).run(until).trajectory_
    crossings = [
        record
        for record in trajectory.events
        if isinstance(record, EventRecord) and record.label == SECTION and record.time >= transient
    ]
    states = np.zeros((len(crossings), problem.states.size))
    for row, record in enumerate(crossings):
        states[row] = record.before
    return Section(np.asarray([record.time for record in crossings], dtype=np.float64), states, problem)


def stroboscopic(
    system: System | Problem,
    period: float,
    *,
    n_points: int = 1000,
    transient: int = 100,
    solver: str | Stepper = 'auto',
    random_state: int | None = None,
) -> Section:
    """Return the states of a periodically forced system at multiples of the forcing period.

    Args:
        system:
            The system, or a problem compiled from one.

        period:
            The period of the forcing, such as `2 * pi / omega`.

        n_points:
            The number of points of the section.

        transient:
            The number of periods discarded before the first point.

        solver:
            The solver, as in `Simulation`.

        random_state:
            The seed of stochastic solvers.

    Returns:
        section:
            The times and states at the sampled periods.

    Raises:
        AnalysisError:
            The system is not a flow.

    Examples:
        >>> import math
        >>> from skcomplex.core.analysis import stroboscopic
        >>> from skcomplex.models.physics.nonlinear_oscillators import duffing
        >>> section = stroboscopic(duffing(), 2 * math.pi / 1.2, n_points=50, transient=10)
        >>> len(section)
        50
    """
    problem = _problem(system)
    until = period * (transient + n_points - 1)
    trajectory = Simulation(problem, solver=solver, dt=period, random_state=random_state).run(until).trajectory_
    return Section(trajectory.times[transient:], trajectory.states[transient:], problem)
