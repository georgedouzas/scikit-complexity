"""Test the `_functions` module of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
SCHEDULED = pytest.mark.skip(
    reason='Engine gap: the functions pass reads phases only from a DiscreteTime schedule, which a System has not.',
)


class CountThenCopyInteraction(Interaction):
    """Interaction that adds one to the count in one phase and copies the count in another."""

    def __init__(
        self,
        label: str,
        *,
        counter: Role,
        count: str | None = None,
        copy: str | None = None,
    ) -> None:
        super().__init__(label)
        self.counter = counter
        self.count = count
        self.copy = copy

    def _prepare(self, context=None):
        """Set the count and the copy, each in its phase."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [
                UpdateLaw(s.counter.count, s.counter.count + 1, phase=self.count),
                UpdateLaw(s.counter.copy, s.counter.count, phase=self.copy),
            ]


def _system(*phases: PhaseSpec, count: str | None = 'count', copy: str | None = 'copy') -> System:
    """Return a counter whose count and copy run in phases of a schedule."""
    interaction = CountThenCopyInteraction('steps', counter='c', count=count, copy=copy)
    settings = {'solver': MesaSolver(phases=list(phases))} if phases else {}
    counter = Element('c', values={'count': 0.0, 'copy': 0.0})
    return System(elements=[counter], interactions=[interaction], **settings)


def _run(system: System, until: int, backend: NumpyBackend | NumbaBackend) -> np.ndarray:
    """Return the count and the copy at every frame of a run."""
    trajectory = system.set_params(backend=backend).prepare().simulate(until=until)
    return np.stack([trajectory.get_values('c.count'), trajectory.get_values('c.copy')], axis=1)


@SCHEDULED
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


@BACKENDS
def test_lower_runs_phases_without_a_schedule_in_the_order_of_their_laws(backend):
    """Test that phases no schedule declares run in the order their laws appear, so the copy reads the new count."""
    system = _system(count=None, copy=None).set_params(steps__count='count', steps__copy='copy')
    values = _run(system, 3, backend)
    assert values.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


@SCHEDULED
def test_lower_runs_the_default_phase_first_and_reversed_phases_in_their_order():
    """Test that laws without a phase run first, and that a schedule of reversed phases copies before it counts."""
    assert _run(_system(PhaseSpec('copy'), count=None), 2, NumpyBackend())[1:].tolist() == [[1.0, 1.0], [2.0, 2.0]]
    reversed_ = _run(_system(PhaseSpec('copy'), PhaseSpec('count')), 2, NumpyBackend())
    assert reversed_[1:].tolist() == [[1.0, 0.0], [2.0, 1.0]]


@SCHEDULED
def test_lower_runs_a_phase_in_the_steps_of_its_period():
    """Test that a phase of period two runs in the even steps alone."""
    values = _run(_system(PhaseSpec('count'), PhaseSpec('copy', period=2)), 4, NumpyBackend())
    assert values[:, 1].tolist() == [0.0, 0.0, 2.0, 2.0, 4.0]


@SCHEDULED
def test_lower_composes_the_equations_of_the_phases():
    """Test that the symbolic map of a later phase reads the values the earlier phases set."""
    problem = _system(PhaseSpec('count'), PhaseSpec('copy')).prepare()._problem
    count = problem.equation_symbols['c.count']
    assert bool(problem.equations['c.copy'] == count + 1)


@SCHEDULED
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
        _system(*phases).prepare()


@SCHEDULED
def test_lower_reads_the_phases_of_a_nested_system():
    """Test that the phases of the schedule of a nested system order its update laws in a system without one."""
    nested = _system(PhaseSpec('count'), PhaseSpec('copy')).set_params(label='counter')
    trajectory = System(systems=[nested]).prepare().simulate(until=1)
    assert trajectory.get_values('counter/c.copy')[1] == 1.0
