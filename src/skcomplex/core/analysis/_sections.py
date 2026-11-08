"""Take sections of a system: Poincaré sections and stroboscopic samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..engine.problems import ProblemSpec
from ..engine.problems._event_kernel import EventKernel
from ..engine.runtime._simulations import Simulation
from ..engine.solvers import Solver
from ..errors import AnalysisError
from ..specs import AnalysisTags, EventRecordSpec
from ..types import Array, DataKind, FormKind
from ._base import _check_choice, _check_settings, _check_tags, _compiled_problem, _composite, _read_system
from .types import Source

KINDS = ('poincare', 'stroboscopic')
SETTINGS = {
    'poincare': {'variables', 'state', 'value', 'direction', 'until', 'transient', 'solver', 'random_state'},
    'stroboscopic': {'variables', 'period', 'n_points', 'transient', 'solver', 'random_state'},
}
"""Settings every kind reads, besides `kind`."""

DIRECTIONS = (-1, 0, 1)
SECTION = '__section'

TAGS = AnalysisTags(DataKind.SYSTEM, time=frozenset({'continuous'}))


@dataclass(frozen=True)
class SectionResult:
    """Section of a system: the states where a flow crosses a value, or at multiples of a period.

    Args:
        problem:
            The compiled problem of a Poincaré or stroboscopic section.

        times:
            The time of every point of a section.

        states:
            One state vector per point of a section.

        values:
            A mapping from the name of every read state to its values at the points of a section, one row per point
            for a vector state.
    """

    problem: ProblemSpec
    times: Array
    states: Array
    values: dict[str, Array]


def section(
    source: Source,
    *,
    kind: str = 'poincare',
    variables: Sequence[str] | None = None,
    state: str | None = None,
    value: float = 0.0,
    direction: int = 1,
    until: float = 1000.0,
    transient: float | None = None,
    period: float | None = None,
    n_points: int = 1000,
    solver: Solver | None = None,
    random_state: int | None = None,
) -> SectionResult:
    """Return the section of a system: the states where a flow crosses a value, or at multiples of a period.

    The kinds run the system:

    - `'poincare'`: the states where the scalar state `state` crosses `value` in a direction, located like an event.
    - `'stroboscopic'`: the states of a periodically forced flow at multiples of the forcing period.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it.

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
            The time before which crossings are discarded, or the number of periods discarded before the first
            stroboscopic point. `None` gives 0 and 100 periods.

        period:
            The period of the forcing, such as `2 * pi / omega`.

        n_points:
            The number of points of a stroboscopic section.

        solver:
            The solver, as in `simulate`.

        random_state:
            The seed of stochastic solvers.

    Returns:
        result:
            The times, the states, and the values of the read states at the points of the section.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a needed setting is missing, the system
            is not a flow in continuous time, or the state is not a scalar state.

        TagError:
            The system does not run in continuous time.

    Examples:
        >>> from skcomplex.core.analysis import section
        >>> from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem
        >>> len(section(HarmonicOscillatorSystem(), state='mass.x', until=20.0).times) > 2
        True
    """
    source = _composite(source)
    _check_choice('section', 'kind', kind, KINDS)
    _check_tags(section, source, kind)
    _check_settings(
        section,
        'kind',
        SETTINGS[kind],
        kind=kind,
        variables=variables,
        state=state,
        value=value,
        direction=direction,
        until=until,
        transient=transient,
        period=period,
        n_points=n_points,
        solver=solver,
        random_state=random_state,
    )
    needed: dict[str, object] = {'state': state} if kind == 'poincare' else {'period': period}
    missing = [name for name, item in needed.items() if item is None]
    if missing:
        error_msg = f'section of kind {kind!r} needs {", ".join(missing)}. Give them.'
        raise AnalysisError(error_msg)
    if direction not in DIRECTIONS:
        error_msg = f'section has direction={direction!r}. Use one of -1, 0, or 1.'
        raise AnalysisError(error_msg)
    problem = _compiled_problem(_read_system(source), 'A section')
    if problem.form not in {FormKind.ODE, FormKind.SDE}:
        error_msg = f'Sections are taken of flows in continuous time, but the system is a {problem.form}.'
        raise AnalysisError(error_msg)
    if kind == 'poincare' and (state is None or not problem.states.has(state) or problem.states.entry(state).size != 1):
        error_msg = f'{state!r} is not a scalar state. Available states are: {", ".join(problem.states.names)}.'
        raise AnalysisError(error_msg)
    if kind == 'poincare':
        times, states = _crossings(
            problem,
            state,
            value=value,
            direction=direction,
            until=until,
            transient=transient,
            solver=solver,
            random_state=random_state,
        )
    else:
        times, states = _samples(
            problem,
            period,
            n_points=n_points,
            transient=transient,
            solver=solver,
            random_state=random_state,
        )
    names = list(problem.states.names) if variables is None else variables
    values = {}
    for name in names:
        read = np.asarray([problem.states.read(row, name) for row in states], dtype=float)
        size = problem.states.entry(name).size
        values[name] = read.reshape(len(times), -1) if size > 1 else read.reshape(-1)
    return SectionResult(problem, times, states, values)


def _crossings(
    problem: ProblemSpec,
    state: str | None,
    *,
    value: float,
    direction: int,
    until: float,
    transient: float | None,
    solver: Solver | None,
    random_state: int | None,
) -> tuple[Array, Array]:
    """Return the times and the states where the state crosses the value of the Poincaré section."""
    if state is None:
        error_msg = 'A Poincaré section needs a state. Give one.'
        raise AnalysisError(error_msg)
    index = problem.states.entry(state).start

    def guard(t: float, x: Array, p: Array) -> float:
        return float(x[index] - value)

    start = transient if transient is not None else 0.0
    event = EventKernel(SECTION, guard, None, int(direction), False).prepare()
    sectioned = dataclasses.replace(problem, events=(*problem.events, event))
    simulation = Simulation(sectioned, solver=solver, dt=until / 100, random_state=random_state)
    trajectory = simulation.prepare().run(until).trajectory_
    crossings = [
        record
        for record in trajectory.events
        if isinstance(record, EventRecordSpec) and record.label == SECTION and record.time >= start
    ]
    states = np.zeros((len(crossings), problem.states.size))
    for row, record in enumerate(crossings):
        states[row] = record.before
    return np.asarray([record.time for record in crossings], dtype=np.float64), states


def _samples(
    problem: ProblemSpec,
    period: float | None,
    *,
    n_points: int,
    transient: float | None,
    solver: Solver | None,
    random_state: int | None,
) -> tuple[Array, Array]:
    """Return the times and the states at multiples of the forcing period."""
    if period is None:
        error_msg = 'A stroboscopic section needs a period. Give one.'
        raise AnalysisError(error_msg)
    step = float(period)
    skipped = int(transient if transient is not None else 100)
    until = step * (skipped + n_points - 1)
    simulation = Simulation(problem, solver=solver, dt=step, random_state=random_state)
    trajectory = simulation.prepare().run(until).trajectory_
    return trajectory.times[skipped:], trajectory.states[skipped:]
