"""Test the `_signals` module of `skcomplex.core.components.laws.utils`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import periodic_guard, pulse_train, series
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


def _rate_system(label, state, rate, backend='auto', size=None, **values):
    """Return one element, or a group of a size, whose state changes at a rate, a function of the namespace."""
    laws = Interaction(
        'rate',
        roles={'target': label},
        laws=lambda s: [RateLaw(getattr(s.target, state).diff(s.t), rate(s))],
    )
    return System(elements=[Element(label, values=values, size=size)], interactions=[laws], backend=backend)


def _driven_by(data, backend='auto'):
    """Return a system whose one rate reads a series input."""
    return _rate_system('d', 'y', lambda s: data(s.t), backend, y=0.0)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_series_compiles_long_held_data(backend):
    """Test that a held series of 1000 points compiles and holds each value until the next time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = _driven_by(lambda t: series(t, times, values, 'previous'), backend).prepare()._problem
    rates = [problem.drift(k + 0.5, problem.initial_state, problem.initial_parameters)[0] for k in range(1000)]
    np.testing.assert_allclose(rates, values, rtol=0, atol=1e-12)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_series_interpolates_long_linear_data(backend):
    """Test that a linear series of 1000 points matches linear interpolation and breaks at every data time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = _driven_by(lambda t: series(t, times, values), backend).prepare()._problem
    probes = np.linspace(-3.0, 1003.0, 2501)
    rates = [problem.drift(time, problem.initial_state, problem.initial_parameters)[0] for time in probes]
    np.testing.assert_allclose(rates, np.interp(probes, times, values), rtol=0, atol=1e-12)
    assert problem.breakpoints.get_between(-1.0, 1000.0) == list(times)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_pulse_train_with_a_count_stops_after_its_last_pulse(backend):
    """Test that a pulse train of three pulses matches the pulses of NumPy and is 0 after the third."""
    problem = _driven_by(lambda t: pulse_train(t, 4, 1, 2.0, start=1, count=3), backend).prepare()._problem
    probes = np.linspace(0.0, 20.0, 401)
    rates = [problem.drift(time, problem.initial_state, problem.initial_parameters)[0] for time in probes]
    expected = np.where((probes >= 1) & (probes < 13) & (np.mod(probes - 1, 4) < 1), 2.0, 0.0)
    np.testing.assert_allclose(rates, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('count', [0, 1.5, True])
def test_pulse_train_rejects_a_count_that_is_not_a_positive_integer(count):
    """Test that a pulse train with a count that is not a positive integer raises an error."""
    with pytest.raises(ValidationError, match='positive whole number'):
        pulse_train(var('t'), 4, 1, count=count)


def test_periodic_guard_crosses_zero_at_the_periodic_times_only():
    """Test that a periodic guard is zero at `start + j period` for `j < count` and nowhere else."""
    t = var('t')
    guard = periodic_guard(t, 2.0, start=1.0, count=3)
    times = np.linspace(-2.0, 12.0, 1401)
    values = np.array([float(guard.subs({t: time})) for time in times])
    zeros = times[np.abs(values) < 1e-9]
    np.testing.assert_allclose(zeros, [1.0, 3.0, 5.0], atol=1e-9)
    signs = np.sign(values[np.abs(values) > 1e-9])
    assert np.count_nonzero(np.diff(signs)) == 3


@pytest.mark.parametrize(('period', 'count'), [(0.0, None), (1.0, 0), (1.0, True)])
def test_periodic_guard_rejects_a_bad_period_or_count(period, count):
    """Test that a periodic guard rejects a period that is not positive and a count that is not a positive integer."""
    with pytest.raises(ValidationError, match='periodic guard'):
        periodic_guard(var('t'), period, count=count)


def _fertility(s):
    """Return the fertility, a table of food per person."""
    return series(s.food, (0, 1, 2), (0.0, 2.0, 3.0))


def test_series_of_a_state_interpolates_and_stays_flat():
    """Test that the fertility, observed as an expression of the element, interpolates the table and holds its ends."""
    fertility = Observable('fertility', _fertility, element='f')
    system = _rate_system('f', 'food', lambda s: 1.0, food=-1.0).prepare([fertility])
    trajectory = system.simulate(until=4, dt=0.5, observables=[fertility])
    food = trajectory.get_values('f.food')
    expected = np.interp(food, [0, 1, 2], [0.0, 2.0, 3.0])
    assert trajectory.get_values('f.fertility') == pytest.approx(expected)
    assert trajectory.get_values('f.fertility')[0] == 0.0
    assert trajectory.get_values('f.fertility')[-1] == 3.0


def test_pulse_train_with_a_count_is_off_after_its_last_pulse_symbolically():
    """Test that a train of two pulses is on in its second pulse and off where a third would be."""
    t = var('t')
    train = pulse_train(t, 10, 1, count=2)
    assert [float(train.subs({t: time})) for time in (10.5, 20.5)] == [1.0, 0.0]


@pytest.mark.parametrize(
    ('points', 'values', 'kind', 'message'),
    [((0, 0), (1, 2), 'linear', 'must increase'), ((0, 1), (1,), 'linear', 'one value'), ((0,), (1,), 'cubic', 'kind')],
)
def test_series_rejects_bad_points_or_kind(points, values, kind, message):
    """Test that points that do not increase, a missing value, or an unknown kind raise an error in plain words."""
    with pytest.raises(ValueError, match=message):
        series(var('t'), points, values, kind)
