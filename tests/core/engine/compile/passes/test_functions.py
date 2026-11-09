"""Test the `_functions` module of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


class CounterElement(Element):
    """Element with a count and a copy of it."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec()] = 0.0,
        copy: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.copy = copy


class CountThenCopyInteraction(Interaction):
    """Interaction that adds one to the count in one phase and copies the count in another."""

    def __init__(
        self,
        label: str,
        *,
        counter: Annotated[str, RoleSpec()],
        count: str | None = None,
        copy: str | None = None,
    ) -> None:
        super().__init__(label)
        self.counter = counter
        self.count = count
        self.copy = copy

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the count and the copy, each in its phase."""
        return [
            UpdateLaw(s.counter.count, s.counter.count + 1, phase=self.count),
            UpdateLaw(s.counter.copy, s.counter.count, phase=self.copy),
        ]


def _system(*phases: PhaseSpec, count: str | None = 'count', copy: str | None = 'copy') -> CompositeSystem:
    """Return a counter whose count and copy run in phases of a schedule."""
    time = DiscreteTime(schedule=PhasesScheduleSpec(phases)) if phases else DiscreteTime()
    interaction = CountThenCopyInteraction('steps', counter='c', count=count, copy=copy)
    return CompositeSystem(elements=[CounterElement('c')], interactions=[interaction], time=time)


def _run(system: CompositeSystem, until: int, backend: NumpyBackend | NumbaBackend) -> np.ndarray:
    """Return the count and the copy at every frame of a run."""
    trajectory = simulate(system, until=until, backend=backend)
    return np.stack([trajectory.values('c.count'), trajectory.values('c.copy')], axis=1)


@BACKENDS
def test_lower_runs_phases_in_order_each_reading_the_one_before(backend):
    """Test that a copy in a later phase reads the count the earlier phase set in the same step."""
    values = _run(_system(PhaseSpec('count'), PhaseSpec('copy')), 3, backend)
    assert values.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


@BACKENDS
def test_lower_without_phases_reads_the_states_at_the_start_of_the_step(backend):
    """Test that update laws without phases read the states at the start of the step, as before phases."""
    values = _run(_system(count=None, copy=None), 3, backend)
    assert values.tolist() == [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]


def test_lower_runs_the_default_phase_first_and_reversed_phases_in_their_order():
    """Test that laws without a phase run first, and that a schedule of reversed phases copies before it counts."""
    assert _run(_system(PhaseSpec('copy'), count=None), 2, NumpyBackend())[1:].tolist() == [[1.0, 1.0], [2.0, 2.0]]
    reversed_ = _run(_system(PhaseSpec('copy'), PhaseSpec('count')), 2, NumpyBackend())
    assert reversed_[1:].tolist() == [[1.0, 0.0], [2.0, 1.0]]


def test_lower_runs_a_phase_in_the_steps_of_its_period():
    """Test that a phase of period two runs in the even steps alone."""
    values = _run(_system(PhaseSpec('count'), PhaseSpec('copy', period=2)), 4, NumpyBackend())
    assert values[:, 1].tolist() == [0.0, 0.0, 2.0, 2.0, 4.0]


def test_lower_composes_the_equations_of_the_phases():
    """Test that the symbolic map of a later phase reads the values the earlier phases set."""
    problem = PassCompiler().prepare(_system(PhaseSpec('count'), PhaseSpec('copy'))).run().problem_
    count = problem.equation_symbols['c.count']
    assert bool(problem.equations['c.copy'] == count + 1)


@pytest.mark.parametrize(
    ('phases', 'match'),
    [
        ((PhaseSpec('count'),), 'which the system does not declare'),
        ((PhaseSpec('count', repeat=2), PhaseSpec('copy')), 'repeats 2 times'),
        ((PhaseSpec('count', activation='simultaneous'), PhaseSpec('copy')), 'take only a period and an offset'),
    ],
)
def test_lower_rejects_undeclared_and_repeated_phases_of_update_laws(phases, match):
    """Test that an update law in an undeclared phase, or a phase that repeats or has an activation, fails."""
    with pytest.raises(ValidationError, match=match):
        PassCompiler().prepare(_system(*phases)).run()


def test_lower_reads_the_phases_of_a_nested_system():
    """Test that the phases of the schedule of a nested system order its update laws in a system without one."""
    nested = _system(PhaseSpec('count'), PhaseSpec('copy'))
    nested.label = 'counter'
    trajectory = simulate(CompositeSystem(systems=[nested]), until=1)
    assert trajectory.values('counter/c.copy')[1] == 1.0
