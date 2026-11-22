"""Test the `_orders` module of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import hyperedges
from skcomplex.core.components.spaces import ContinuousSpace, RadiusNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class _Drag(Interaction):
    """Interaction whose law differentiates a product in time, `(m v)' = -c v`, or the speed itself."""

    def __init__(self, label: str, *, target: Role = 'body', c: float = 1.0, reads_speed: bool = False) -> None:
        super().__init__(label)
        self.target = target
        self.c = c
        self.reads_speed = reads_speed

    def _prepare(self, context: SimpleNamespace) -> None:
        super()._prepare(context)
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


def _velocity_alignment(s):
    """Return the pull of the velocity of every member to that of the other member of each pair within reach."""
    x, other = s.target.x, s.target.others[0].x
    pull = hyperedges(other.diff(s.t) - x.diff(s.t), size=2, over=RadiusNeighborhood(2.0))
    return [RateLaw(x.diff(s.t, 2), pull)]


def test_hyperedges_read_the_derivatives_of_their_other_members():
    """Test that `others[0].x.diff(t)` in a reduction over pairs of neighbours reads the velocity of the other."""
    birds = Element(
        'birds',
        values={'x': np.array([[0.0, 0.0], [1.0, 0.0]]), "x'": np.array([[1.0, 0.0], [-1.0, 0.0]])},
        size=2,
        space=ContinuousSpace(n_dim=2),
    )
    align = Interaction('align', roles={'target': 'birds'}, laws=_velocity_alignment)
    trajectory = System(elements=[birds], interactions=[align]).prepare().simulate(until=0.01, dt=0.01)
    velocities = trajectory.get_values("birds.x'")[-1]
    assert velocities[0, 0] == pytest.approx(1.0 - 2.0 * 0.01, abs=1e-3)
    assert velocities[1, 0] == pytest.approx(-1.0 + 2.0 * 0.01, abs=1e-3)
