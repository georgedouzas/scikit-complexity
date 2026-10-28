"""Test the `_delays` module of `skcomplex.core.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import distributed_delay, step
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation


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
        .check()
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
    trajectory = Simulation(System(elements=[DelayedElement('d', kind=kind)]), dt=1).check().run(80).trajectory_
    hidden = np.atleast_2d(trajectory.values(f'd.{kind}_0').T).T
    assert hidden.shape[1] == stages
    level = 6.0 / stages if kind == 'material_delay1' else 1.0
    assert hidden[-1, -1] == pytest.approx(level, rel=1e-3)
