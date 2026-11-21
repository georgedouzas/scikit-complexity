"""Test the `_delays` module of `skcomplex.core.components.laws.utils`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var
from scipy import stats

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import distributed_delay, step
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.systems import System

STEP_RESPONSES = {
    'material1': (lambda value, time: distributed_delay(value, time, 1), lambda t: 1 - np.exp(-t / 2)),
    'material3': (lambda value, time: distributed_delay(value, time, 3), lambda t: stats.gamma(3, scale=2 / 3).cdf(t)),
    'smoothing1': (
        lambda value, time: distributed_delay(value, time, conserved=False),
        lambda t: 1 - np.exp(-t / 2),
    ),
}


def _system(rate, backend='auto', **values):
    """Return the output stock `d.out` that integrates a rate, a function of the namespace of the stock."""
    integrate = Interaction(
        'integrate',
        roles={'target': 'd'},
        laws=lambda s: [RateLaw(s.target.out.diff(s.t), rate(s))],
    )
    return System(elements=[Element('d', values={'out': 0.0, **values})], interactions=[integrate], backend=backend)


def _delayed(kind):
    """Return the output stock of a delay of a unit step at time 1 with delay time 6."""
    conserved, order = kind.startswith('material'), int(kind[-1])
    return _system(lambda s: distributed_delay(step(s.t, 1.0), s.target.T, order=order, conserved=conserved), T=6.0)


def test_third_order_delay_follows_the_erlang_step_response():
    """Test that the output of a third-order delay of a step reaches half the step 5.35 after it."""
    trajectory = _delayed('material_delay3').prepare().simulate(until=14, dt=0.01)
    times = np.asarray(trajectory.times)
    output = trajectory.get_values('integrate.material_delay3_0')[:, 2] / 2.0
    assert times[np.argmax(output >= 0.5)] - 1.0 == pytest.approx(stats.gamma(3, scale=2).median(), abs=0.02)
    assert stats.gamma(3, scale=2).median() == pytest.approx(5.35, abs=0.01)


@pytest.mark.parametrize(('kind', 'stages'), [('material_delay1', 1), ('smoothing_delay1', 1), ('smoothing_delay3', 3)])
def test_delays_and_smoothings_add_hidden_stocks(kind, stages):
    """Test that delays and smoothings add one hidden state per stage that tends to the step."""
    trajectory = _delayed(kind).prepare().simulate(until=80, dt=1)
    hidden = np.atleast_2d(trajectory.get_values(f'integrate.{kind}_0').T).T
    assert hidden.shape[1] == stages
    level = 6.0 / stages if kind == 'material_delay1' else 1.0
    assert hidden[-1, -1] == pytest.approx(level, rel=1e-3)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
@pytest.mark.parametrize('name', list(STEP_RESPONSES))
def test_stock_and_flow_delays_follow_their_step_responses(name, backend):
    """Test that the delays of a step rise as their step responses in NumPy and SciPy."""
    delay, response = STEP_RESPONSES[name]
    system = _system(lambda s: delay(step(s.t, 1.0), 2.0), backend)
    trajectory = system.prepare().simulate(until=9, dt=0.001)
    times = np.asarray(trajectory.times)
    rates = np.gradient(trajectory.get_values('d.out'), times)
    later = times > 1.5
    np.testing.assert_allclose(rates[later], response(times[later] - 1.0), atol=2e-3)


@pytest.mark.parametrize(
    ('order', 'conserved', 'expected'),
    [(1, True, 'material_delay1(x, 4)'), (3, True, 'material_delay3(x, 4)'), (1, False, 'smoothing_delay1(x, 4)')],
)
def test_stock_and_flow_delays_are_distributed_delays(order, conserved, expected):
    """Test that the stock-and-flow delays are distributed delays of their order and kind."""
    assert str(distributed_delay(var('x'), 4, order, conserved=conserved)) == expected


@pytest.mark.parametrize(('order', 'time', 'message'), [(0, 4, 'positive whole number'), (1, -1, 'must be positive')])
def test_distributed_delay_rejects_a_bad_order_or_time(order, time, message):
    """Test that an order below 1 or a delay time that is not positive raises an error in plain words."""
    with pytest.raises(ValueError, match=message):
        distributed_delay(var('x'), time, order)
