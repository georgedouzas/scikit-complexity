"""Test the `_minimize` module of `skcomplex.core.engine.solvers`."""

from types import SimpleNamespace

import pytest
from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, ConstraintLaw, ObjectiveLaw
from skcomplex.core.engine.solvers import MinimizeSolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class PointElement(Element):
    """Point on a line."""

    def __init__(self, label: str, *, x: float = 0.0):
        super().__init__(label)
        self.x = x


class PullInteraction(Interaction):
    """Interaction that pulls a point towards a goal with a quadratic cost."""

    def __init__(self, label: str, *, target: Role = 'a', goal: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.goal = goal

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the quadratic cost."""
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [ObjectiveLaw((context.target.x - self.goal) ** 2, context.target.x)]


class BudgetInteraction(Interaction):
    """Interaction that minimises the squared sizes of two points whose sum is one."""

    def __init__(
        self,
        label: str,
        *,
        a: Role = 'a',
        b: Role = 'b',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the cost and the budget."""
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [
            ObjectiveLaw(context.a.x**2 + context.b.x**2, [context.a.x, context.b.x]),
            ConstraintLaw(context.a.x + context.b.x == 1, target=context.b.x),
        ]


class ImplicitInteraction(Interaction):
    """Interaction whose constraint has no explicit solution."""

    def __init__(
        self,
        label: str,
        *,
        a: Role = 'a',
        b: Role = 'b',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the cost and an implicit constraint."""
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [
            ObjectiveLaw((context.a.x - 1) ** 2, context.a.x),
            ConstraintLaw(context.b.x == cos(context.b.x) + context.a.x, target=context.b.x),
        ]


def points(*values: float) -> list[PointElement]:
    """Return points labelled `a`, `b`, and so on, at the given positions."""
    return [PointElement(label, x=value) for label, value in zip('abc', values, strict=False)]


def test_prepare_finds_an_unconstrained_minimum():
    """A quadratic cost is minimised at its goal, and the minimiser is the default solver."""
    system = System(elements=points(5.0), interactions=[PullInteraction('pull', goal=2.0)]).prepare()
    assert isinstance(system.solver_, MinimizeSolver)
    assert system.simulate(until=1.0).get_values('a.x')[0] == pytest.approx(2.0, abs=1e-6)


def test_prepare_adds_objectives_of_several_interactions():
    """The objectives of several interactions add up over several elements."""
    pulls = [
        PullInteraction('first', target='a', goal=1.0),
        PullInteraction('second', target='a', goal=3.0),
        PullInteraction('third', target='b', goal=-3.0),
    ]
    trajectory = System(elements=points(0.0, 0.0), interactions=pulls).prepare().simulate(until=1.0)
    assert trajectory.get_values('a.x')[0] == pytest.approx(2.0, abs=1e-6)
    assert trajectory.get_values('b.x')[0] == pytest.approx(-3.0, abs=1e-6)


def test_prepare_holds_an_explicit_constraint_at_the_minimum():
    """A constraint solved explicitly holds at the minimum."""
    system = System(elements=points(3.0, -1.0), interactions=[BudgetInteraction('budget')]).prepare()
    trajectory = system.simulate(until=1.0)
    assert trajectory.get_values('a.x')[0] == pytest.approx(0.5, abs=1e-6)
    assert trajectory.get_values('b.x')[0] == pytest.approx(0.5, abs=1e-6)


def test_prepare_holds_an_implicit_constraint_at_the_minimum():
    """A constraint without an explicit solution holds at the minimum as an equality constraint."""
    system = System(elements=points(0.0, 0.5), interactions=[ImplicitInteraction('implicit')]).prepare()
    trajectory = system.simulate(until=1.0)
    a, b = trajectory.get_values('a.x')[0], trajectory.get_values('b.x')[0]
    assert a == pytest.approx(1.0, abs=1e-5)
    assert b == pytest.approx(float(cos(b)) + a, abs=1e-6)
