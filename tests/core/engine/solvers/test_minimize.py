"""Test the `_minimize` module of `skcomplex.core.engine.solvers`."""

from types import SimpleNamespace
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw
from skcomplex.core.engine.solvers import MinimizeSolver
from skcomplex.core.specs import RoleSpec, StateSpec, VisualSpec
from skcomplex.core.systems import System


class PointElement(Element):
    """Point on a line."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


class PullInteraction(Interaction):
    """Interaction that pulls a point towards a goal with a quadratic cost."""

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()] = 'a', goal: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.goal = goal

    def _get_laws(self, s: SimpleNamespace) -> list[BaseComponent]:
        """Return the quadratic cost."""
        return [ObjectiveLaw((s.target.x - self.goal) ** 2, s.target.x)]


class BudgetInteraction(Interaction):
    """Interaction that minimises the squared sizes of two points whose sum is one."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[str, RoleSpec()] = 'a',
        b: Annotated[str, RoleSpec()] = 'b',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseComponent]:
        """Return the cost and the budget."""
        return [
            ObjectiveLaw(s.a.x**2 + s.b.x**2, [s.a.x, s.b.x]),
            ConstraintLaw(s.a.x + s.b.x == 1, target=s.b.x),
        ]


class ImplicitInteraction(Interaction):
    """Interaction whose constraint has no explicit solution."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[str, RoleSpec()] = 'a',
        b: Annotated[str, RoleSpec()] = 'b',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseComponent]:
        """Return the cost and an implicit constraint."""
        return [
            ObjectiveLaw((s.a.x - 1) ** 2, s.a.x),
            ConstraintLaw(s.b.x == cos(s.b.x) + s.a.x, target=s.b.x),
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
