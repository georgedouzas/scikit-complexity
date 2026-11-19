"""Test the `_event` module of `skcomplex.core.components.laws`."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, EventLaw, RateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class BounceInteraction(Interaction):
    """Fall and bounce of a ball, with the guard and the reset given by the argument `law`."""

    def __init__(self, label: str, *, target: Role, law: str = 'symbols') -> None:
        super().__init__(label)
        self.target = target
        self.law = law

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the fall and the bounce."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            name, speed = f'{self.target}.y', f'{self.target}.v'
            crossings = {
                'symbols': lambda: EventLaw(s.y, {s.v: -0.5 * s.v}, direction=-1),
                'functions': lambda: EventLaw(
                    lambda t, values: values[name],
                    {s.v: lambda t, values: -0.5 * values[speed]},
                    -1,
                ),
                'relative': lambda: EventLaw(
                    CallSpec(lambda t, values: values['target.y'], relative=True),
                    {s.v: CallSpec(lambda t, values: -0.5 * values['target.v'], relative=True)},
                    -1,
                ),
                'mixed': lambda: EventLaw(lambda t, values: values[name], {s.v: -0.5 * s.v}, -1),
                'arguments': lambda: EventLaw(CallSpec(lambda t, y: y, (s.y,)), direction=-1),
            }
            self.laws_ = [RateLaw(s.y, s.v), RateLaw(s.v, -9.81), crossings[self.law]()]


def _ball(law, backend='auto'):
    """Return the system of a ball whose crossing has the given form."""
    return System(
        elements=[Element('ball', values={'y': 1.0, 'v': 0.0})],
        interactions=[BounceInteraction('bounce', target='ball', law=law)],
        backend=backend,
    )


def _trajectory(law, backend=None):
    """Return the trajectory of a ball whose crossing has the given form."""
    return _ball(law, backend or NumpyBackend()).prepare().simulate(until=1.0, dt=0.001)


@pytest.mark.parametrize('law', ['functions', 'relative', 'mixed'])
def test_crossing_with_functions_equals_the_symbolic_crossing(law):
    """Test that a crossing with a function fires where the symbolic crossing fires and applies the same reset."""
    reference, trajectory = _trajectory('symbols'), _trajectory(law)
    assert [event.time for event in trajectory.events] == pytest.approx([event.time for event in reference.events])
    assert trajectory.get_values('ball.y') == pytest.approx(reference.get_values('ball.y'), abs=1e-9)


def test_crossing_with_functions_on_numba_runs_in_python():
    """Test that a crossing with a function runs on the Numba backend and equals the NumPy run."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        trajectory = _trajectory('functions', NumbaBackend())
    assert trajectory.get_values('ball.y') == pytest.approx(_trajectory('symbols').get_values('ball.y'), abs=1e-9)


def test_crossing_with_a_function_reads_every_member_of_a_group():
    """Test that a guard over a group gives one value per member, and each member fires alone."""

    def laws(s):
        """Return the fall and the reset of one member."""
        guard = CallSpec(lambda t, values: values['target.y'], relative=True)
        return [RateLaw(s.target.y, -1.0), EventLaw(guard, {s.target.y: lambda t, values: 5.0}, -1)]

    herd = Element('herd', values={'y': [1.0, 2.0]}, size=2)
    system = System(elements=[herd], interactions=[Interaction('fall', roles={'target': 'herd'}, laws=laws)])
    trajectory = system.prepare().simulate(until=1.5, dt=0.01)
    assert np.asarray(trajectory.get_values('herd.y'))[-1] == pytest.approx([4.5, 0.5], abs=1e-6)


def test_crossing_rejects_a_call_with_arguments():
    """Test that a crossing names the forms of a function it takes."""
    with pytest.raises(ValidationError, match='An event takes'):
        _ball('arguments').prepare().simulate(until=0, dt=0.01)
