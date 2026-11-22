"""Test the `_history` module of `skcomplex.core.components.laws.utils`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import abs_symbolic, cos, sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import residence, where
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

from ..conftest import PRECISE, MassElement


def _run(element, laws, until, *, dt=None, solver='auto'):
    """Return the run of one element under the laws of one interaction, a function of the namespace."""
    rule = Interaction('laws', roles={'target': element.label}, laws=laws)
    return System(elements=[element], interactions=[rule], solver=solver).prepare().simulate(until=until, dt=dt)


def _double_pendulum(s):
    """Return the coupled equations of the double pendulum."""
    a, b, g, t = s.target.a, s.target.b, s.target.g, s.t
    return [
        ConstraintLaw(
            2 * a.diff(t, 2) + cos(a - b) * b.diff(t, 2) == -(b.diff(t) ** 2) * sin(a - b) - 2 * g * sin(a),
        ),
        ConstraintLaw(b.diff(t, 2) + cos(a - b) * a.diff(t, 2) == a.diff(t) ** 2 * sin(a - b) - g * sin(b)),
    ]


def _jerk(s):
    """Return Sprott's jerk equation."""
    x, t = s.target.x, s.t
    return [RateLaw(x.diff(t, 3), -s.target.A * x.diff(t, 2) - x.diff(t) + abs_symbolic(x) - 1)]


def test_coupled_equations_conserve_energy():
    """Test that the double pendulum's coupled equations are solved jointly and conserve its energy."""
    pendulum = Element('dp', values={'a': 1.0, 'b': 0.5, 'g': 9.81})
    trajectory = _run(pendulum, _double_pendulum, 10, dt=0.05, solver=PRECISE)
    a, b = trajectory.get_values('dp.a'), trajectory.get_values('dp.b')
    a_d1, b_d1 = trajectory.get_values("dp.a'"), trajectory.get_values("dp.b'")
    energy = a_d1**2 + 0.5 * b_d1**2 + a_d1 * b_d1 * np.cos(a - b) - 2 * 9.81 * np.cos(a) - 9.81 * np.cos(b)
    assert np.abs(energy - energy[0]).max() < 1e-7 * abs(energy[0])


def test_third_order_jerk_is_bounded_and_chaotic():
    """Test that Sprott's jerk equation compiles to three equations and gives a bounded, sensitive trajectory."""
    first = _run(Element('j', values={'x': 0.0, "x''": 1.0, 'A': 0.6}), _jerk, 300, dt=0.5, solver=PRECISE)
    assert first.problem.states.names == ['j.x', 'j.x_d1', 'j.x_d2']
    second = _run(Element('j', values={'x': 1e-8, "x''": 1.0, 'A': 0.6}), _jerk, 300, dt=0.5, solver=PRECISE)
    x = first.get_values('j.x')
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10
    assert np.abs(x - second.get_values('j.x')).max() > 0.1


def test_derivative_at_order_is_rejected_in_rates():
    """Test that a rate cannot read the derivative of a state's own order."""
    with pytest.raises(ValidationError, match='below its declared order'):
        _run(MassElement('mass'), lambda s: [RateLaw(s.target.x.diff(s.t, 2), s.target.x.diff(s.t, 2))], 1)


def test_residence_counts_the_steps_a_state_holds_its_value():
    """Test that the residence is 0 in the step a state takes a value and grows by 1 every step it keeps it."""
    toggles = Element('toggles', values={'x': 0.0, 'held': [2.0, 1.0]}, size=2)

    def switch(s):
        """Return the switch of the value after it has held it for `held` steps."""
        x = s.target.x
        return [UpdateLaw(x, where(residence(x) >= s.target.held, 1 - x, x))]

    trajectory = _run(toggles, switch, 6)
    np.testing.assert_array_equal(
        trajectory.get_values('toggles.x').T,
        [[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1]],
    )


def test_residence_outside_update_laws_is_rejected():
    """Test that a law other than an update law cannot read the residence of a state."""
    with pytest.raises(ValidationError, match='only update laws'):
        _run(MassElement('mass'), lambda s: [RateLaw(s.target.x, residence(s.target.x))], 1)
