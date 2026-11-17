"""Test the `_frame_spec` module of `skcomplex.core.engine.problems`."""

from types import SimpleNamespace

import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.runtime import BaseObserver
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class _Swinging(Interaction):
    """Interaction of a pendulum, `phi'' = -sin(phi)`."""

    def __init__(self, label: str, *, target: Role = 'swing') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            phi, t = context.target.phi, context.t
            self.laws_ = [RateLaw(phi.diff(t, 2), -sin(phi))]


class _SpeedObserver(BaseObserver):
    """Observer that keeps the angular speed of the swing at every frame."""

    def _prepare(self, context):
        self.speeds_ = []

    def _run(self, inputs):
        if inputs is not None:
            self.speeds_.append(float(inputs.get_value("swing.phi'")))


def test_get_value_reads_a_derivative_by_its_name_with_a_prime():
    """Test that an observer reads the angular speed of a pendulum as `swing.phi'`."""
    swing = Element('swing', values={'phi': 0.0, "phi'": 0.5})
    observer = _SpeedObserver()
    system = System(elements=[swing], interactions=[_Swinging('swinging')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5, observers=[observer])
    assert observer.speeds_ == pytest.approx(trajectory.get_values("swing.phi'").tolist())
    assert observer.speeds_[0] == pytest.approx(0.5)
