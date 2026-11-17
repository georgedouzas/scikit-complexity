"""Test the `_orders` module of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class _Drag(Interaction):
    """Interaction whose law differentiates a product in time, `(m v)' = -c v`, or the speed itself."""

    def __init__(self, label: str, *, target: Role = 'body', c: float = 1.0, reads_speed: bool = False) -> None:
        super().__init__(label)
        self.target = target
        self.c = c
        self.reads_speed = reads_speed

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            body, t = context.target, context.t
            if self.reads_speed:
                self.laws_ = [RateLaw(body.v.diff(t), -body.v.diff(t))]
            else:
                self.laws_ = [RateLaw(body.v.diff(t), -context.c * body.v / body.m + body.m.diff(t))]


def test_derivative_of_a_parameter_is_zero():
    """Test that the derivative of a constant mass is zero, so the speed decays at the rate `c / m`."""
    body = Element('body', values={'v': 1.0, 'm': 2.0})
    system = System(elements=[body], interactions=[_Drag('drag')]).prepare()
    assert system.equations_ == {'body.v': '-drag.c*body.v/body.m'}
    speeds = system.simulate(until=1.0, dt=1.0).get_values('body.v')
    assert speeds[-1] == pytest.approx(np.exp(-0.5), rel=1e-4)


def test_derivative_of_a_state_at_its_order_raises_a_clear_error():
    """Test that a rate law that reads the derivative it gives names the order and how to raise it."""
    body = Element('body', values={'v': 1.0, 'm': 2.0})
    system = System(elements=[body], interactions=[_Drag('drag', reads_speed=True)])
    with pytest.raises(ValidationError, match=r'only below its declared order, here 1'):
        system.prepare()
