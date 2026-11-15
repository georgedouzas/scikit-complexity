"""Test the `_state_inference` module of `skcomplex.core.engine.compile`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import abs_symbolic, oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Particle(Element):
    """Particle with a position and a mass, declared by nothing but its arguments."""

    def __init__(self, label: str, x: float = 0.0, m: float = 1.0) -> None:
        super().__init__(label)
        self.x = x
        self.m = m


class Herd(Element):
    """Herd whose size is nonnegative and whose status is one of two categories."""

    def __init__(
        self,
        label: str,
        n: Annotated[float, RealSet.closed_open(0, oo)] = 40.0,
        status: Annotated[str, {'S', 'I'}] = 'S',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.status = status


class Economy(Element):
    """Economy with an income."""

    def __init__(self, label: str, y: float = 1.0) -> None:
        super().__init__(label)
        self.y = y


class Gravity(Interaction):
    """Interaction that gives a constant acceleration to its target."""

    def __init__(self, label: str, *, target: Role = 'ball', g: float = 9.81) -> None:
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set the acceleration as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(t, 2), -context.g)]


class Jerk(Interaction):
    """Interaction of Sprott's jerk equation."""

    def __init__(self, label: str, *, target: Role = 'ball', A: float = 0.6) -> None:
        super().__init__(label)
        self.target = target
        self.A = A

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set the third derivative as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            jerk = -context.A * x.diff(t, 2) - x.diff(t) + abs_symbolic(x) - 1
            self.laws_ = [RateLaw(x.diff(t, 3), jerk)]


class Mixed(Interaction):
    """Interaction that gives two orders of one state."""

    def __init__(self, label: str, *, target: Role = 'ball') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set laws of two orders."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x, 1.0), RateLaw(x.diff(t, 2), 1.0)]


class Accelerator(Interaction):
    """Interaction of Samuelson's multiplier and accelerator."""

    def __init__(self, label: str, *, target: Role = 'economy', a: float = 1.2, b: float = -0.5) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set the next income as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            y, t = context.target.y, context.t
            self.laws_ = [UpdateLaw(y, 1 + context.a * y(t=t - 1) + context.b * y(t=t - 2))]


class Feedback(Interaction):
    """Interaction of a delayed negative feedback."""

    def __init__(self, label: str, *, target: Role = 'ball', tau: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.tau = tau

    def _prepare(self, context: SimpleNamespace | None = None) -> None:
        """Set the delayed decay as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x, -x(t=t - context.tau))]


def test_get_declared_makes_untouched_values_parameters():
    """A particle no law acts on has no states, and its values are parameters."""
    system = System(elements=[Particle('ball', x=2.0)]).prepare()
    assert system.states_ == ()
    assert system.parameters_ == ('ball.m', 'ball.x')


def test_get_declared_reads_the_order_from_the_laws():
    """A second derivative in a law makes the position a state of order two, named with primes."""
    system = System(elements=[Particle('ball', x=10.0)], interactions=[Gravity('gravity')]).prepare()
    assert system.states_ == ('ball.x', "ball.x'")
    assert system.parameters_ == ('ball.m', 'gravity.g')
    assert system.equations_ == {'ball.x': "ball.x'", "ball.x'": '-gravity.g'}
    trajectory = system.simulate(until=1.4, dt=0.01)
    exact = 10.0 - 9.81 * trajectory.times**2 / 2
    assert trajectory.get_values('ball.x') == pytest.approx(exact, abs=1e-9)


def test_get_declared_starts_from_a_given_derivative():
    """The initial value of a derivative is given by its name with a prime."""
    system = System(elements=[Particle('ball')], interactions=[Gravity('gravity')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5, initial_state={"ball.x'": 1.5})
    assert trajectory.get_values("ball.x'")[0] == pytest.approx(1.5)


def test_get_declared_reads_a_third_order():
    """Sprott's jerk equation compiles to three first-order states, and its trajectory stays bounded."""
    system = System(elements=[Particle('ball', x=0.02)], interactions=[Jerk('jerk')]).prepare()
    assert system.states_ == ('ball.x', "ball.x'", "ball.x''")
    assert np.abs(system.simulate(until=50.0, dt=1.0).get_values('ball.x')).max() < 10


def test_get_declared_reads_lags():
    """Lags in a law keep the past values of a state, named with indices."""
    system = System(elements=[Economy('economy')], interactions=[Accelerator('samuelson')]).prepare()
    assert system.states_ == ('economy.y', 'economy.y[-1]', 'economy.y[-2]')
    assert system.equations_['economy.y[-1]'] == 'economy.y'


def test_get_declared_rejects_two_orders():
    """Laws that give two orders of one state raise an error."""
    with pytest.raises(ValidationError, match='order'):
        System(elements=[Particle('ball')], interactions=[Mixed('mixed')]).prepare()


@pytest.mark.parametrize(('values', 'message'), [({'n': -1.0}, r'herd\.n = -1\.0'), ({'status': 'X'}, 'herd.status')])
def test_get_declared_checks_allowed_values(values, message):
    """A value outside the values its annotation allows raises an error that names it."""
    with pytest.raises(ValidationError, match=message):
        System(elements=[Herd('herd', **values)]).prepare()


def test_get_declared_reads_a_delay():
    """A state read a delay before the current time follows the delay equation."""
    system = System(elements=[Particle('ball', x=1.0)], interactions=[Feedback('feedback')]).prepare()
    assert system.equations_ == {'ball.x': '-ball.x(t - feedback.tau)'}
    values = system.simulate(until=2.0, dt=1.0).get_values('ball.x')
    assert values == pytest.approx([1.0, 0.0, -0.5], abs=1e-6)
