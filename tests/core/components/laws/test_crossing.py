"""Test the `_crossing` module of `skcomplex.core.components.laws`."""

import warnings
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import CrossingLaw, RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.specs import CallSpec, StateSpec, VisualSpec


class BallElement(Element):
    """Ball that falls and bounces, with the guard and the reset given by the argument `law`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        law: str = 'symbols',
        y: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.law = law
        self.y = y
        self.v = v

    def _get_laws(self, s):
        """Return the fall and the bounce."""
        name = 'ball.y' if self.label == 'ball' else f'{self.label}.y'
        speed = name.replace('.y', '.v')
        crossings = {
            'symbols': CrossingLaw(s.y, {s.v: -0.5 * s.v}, direction=-1),
            'functions': CrossingLaw(lambda t, values: values[name], {s.v: lambda t, values: -0.5 * values[speed]}, -1),
            'relative': CrossingLaw(
                CallSpec(lambda t, values: values['y'], relative=True),
                {s.v: CallSpec(lambda t, values: -0.5 * values['v'], relative=True)},
                -1,
            ),
            'mixed': CrossingLaw(lambda t, values: values[name], {s.v: -0.5 * s.v}, -1),
            'arguments': CrossingLaw(CallSpec(lambda t, y: y, (s.y,)), direction=-1),
        }
        return [RateLaw(s.y, s.v), RateLaw(s.v, -9.81), crossings[self.law]]


def _trajectory(law, backend=None):
    """Return the trajectory of a ball whose crossing has the given form."""
    system = CompositeSystem(elements=[BallElement('ball', law=law)])
    return simulate(system, until=1.0, dt=0.001, backend=backend or NumpyBackend())


@pytest.mark.parametrize('law', ['functions', 'relative', 'mixed'])
def test_crossing_with_functions_equals_the_symbolic_crossing(law):
    """Test that a crossing with a function fires where the symbolic crossing fires and applies the same reset."""
    reference, trajectory = _trajectory('symbols'), _trajectory(law)
    assert [event.time for event in trajectory.events] == pytest.approx([event.time for event in reference.events])
    assert trajectory.values('ball.y') == pytest.approx(reference.values('ball.y'), abs=1e-9)


def test_crossing_with_functions_on_numba_runs_in_python():
    """Test that a crossing with a function runs on the Numba backend and equals the NumPy run."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        trajectory = _trajectory('functions', NumbaBackend())
    assert trajectory.values('ball.y') == pytest.approx(_trajectory('symbols').values('ball.y'), abs=1e-9)


def test_crossing_with_a_function_reads_every_member_of_a_group():
    """Test that a guard over a group gives one value per member, and each member fires alone."""

    class FallingElement(Element):
        """Falling member."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

        def _get_laws(self, s):
            """Return the fall and the reset of one member."""
            guard = CallSpec(lambda t, values: values['y'], relative=True)
            return [RateLaw(s.y, -1.0), CrossingLaw(guard, {s.y: lambda t, values: 5.0}, -1)]

    group = GroupElement('herd', element=FallingElement, size=2, values={'y': [1.0, 2.0]})
    trajectory = simulate(CompositeSystem(elements=[group]), until=1.5, dt=0.01)
    assert np.asarray(trajectory.values('herd.y'))[-1] == pytest.approx([4.5, 0.5], abs=1e-6)


def test_crossing_rejects_a_call_with_arguments():
    """Test that a crossing names the forms of a function it takes."""
    system = CompositeSystem(elements=[BallElement('ball', law='arguments')])
    with pytest.raises(ValidationError, match='An event takes'):
        simulate(system, until=0, dt=0.01)
