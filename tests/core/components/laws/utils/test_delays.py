"""Test the `_delays` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var
from scipy import stats

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import delay1, delay3, distributed_delay, smooth, step
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class DelayedElement(Element):
    """Output stock that integrates a delay of a unit step at time 1 with delay time 6."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kind: str = 'material_delay3',
        out: Annotated[object, StateSpec()] = 0.0,
        T: Annotated[object, ParameterSpec()] = 6.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.out = out
        self.T = T

    def _get_laws(self, s):
        """Return the integration of the delayed step."""
        conserved, order = self.kind.startswith('material'), int(self.kind[-1])
        return [RateLaw(s.out, distributed_delay(step(s.t, 1.0), s.T, order=order, conserved=conserved))]


def test_third_order_delay_follows_the_erlang_step_response():
    """Test that the output of a third-order delay of a step reaches half the step 5.35 after it."""
    trajectory = (
        Simulation(System(elements=[DelayedElement('d', kind='material_delay3')]), dt=0.01)
        .prepare()
        .run(14)
        .trajectory_
    )
    times = np.asarray(trajectory.times)
    output = trajectory.values('d.material_delay3_0')[:, 2] / 2.0
    assert times[np.argmax(output >= 0.5)] - 1.0 == pytest.approx(stats.gamma(3, scale=2).median(), abs=0.02)
    assert stats.gamma(3, scale=2).median() == pytest.approx(5.35, abs=0.01)


@pytest.mark.parametrize(('kind', 'stages'), [('material_delay1', 1), ('smoothing_delay1', 1), ('smoothing_delay3', 3)])
def test_delays_and_smoothings_add_hidden_stocks(kind, stages):
    """Test that delays and smoothings add one hidden state per stage that tends to the step."""
    trajectory = Simulation(System(elements=[DelayedElement('d', kind=kind)]), dt=1).prepare().run(80).trajectory_
    hidden = np.atleast_2d(trajectory.values(f'd.{kind}_0').T).T
    assert hidden.shape[1] == stages
    level = 6.0 / stages if kind == 'material_delay1' else 1.0
    assert hidden[-1, -1] == pytest.approx(level, rel=1e-3)


class StepResponseElement(Element):
    """Output state whose rate is a named delay of a unit step at time 1 with delay time 2."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        delay: object = None,
        out: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.delay = delay
        self.out = out

    def _get_laws(self, s):
        """Return the integration of the delayed step."""
        return [RateLaw(s.out, self.delay(step(s.t, 1.0), 2.0))]


STEP_RESPONSES = {
    'delay1': (delay1, lambda t: 1 - np.exp(-t / 2)),
    'delay3': (delay3, lambda t: stats.gamma(3, scale=2 / 3).cdf(t)),
    'smooth': (smooth, lambda t: 1 - np.exp(-t / 2)),
}


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
@pytest.mark.parametrize('name', list(STEP_RESPONSES))
def test_stock_and_flow_delays_follow_their_step_responses(name, backend):
    """Test that the delays of a step rise as their step responses in NumPy and SciPy."""
    delay, response = STEP_RESPONSES[name]
    system = System(elements=[StepResponseElement('d', delay=delay)])
    trajectory = Simulation(system, dt=0.001, backend=backend).prepare().run(9).trajectory_
    times = np.asarray(trajectory.times)
    rates = np.gradient(trajectory.values('d.out'), times)
    later = times > 1.5
    np.testing.assert_allclose(rates[later], response(times[later] - 1.0), atol=2e-3)


@pytest.mark.parametrize(
    ('delay', 'expected'),
    [(delay1, 'material_delay1(x, 4)'), (delay3, 'material_delay3(x, 4)'), (smooth, 'smoothing_delay1(x, 4)')],
)
def test_stock_and_flow_delays_are_distributed_delays(delay, expected):
    """Test that the stock-and-flow delays are distributed delays of their order and kind."""
    assert str(delay(var('x'), 4)) == expected
