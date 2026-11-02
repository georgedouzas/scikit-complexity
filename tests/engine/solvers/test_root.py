"""Test the static form and the solver of its equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, exp, var
from scipy.special import logsumexp

from skcomplex.core.base import BasePart
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import EquationLaw, RateLaw
from skcomplex.core.laws.utils import log_sum_exp, total
from skcomplex.core.spaces import StaticTime
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import SolverError
from skcomplex.specs import IssueSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec
from skcomplex.types import FormKind


class SurvivalElement(Element):
    """Survival at a dose."""

    def __init__(
        self,
        label: str,
        S: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 0.3,
        beta: Annotated[float, ParameterSpec()] = 0.03,
        D: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.S = S
        self.alpha = alpha
        self.beta = beta
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the survival equation."""
        return [EquationLaw(exp(-s.alpha * s.D - s.beta * s.D**2) == s.S, target=s.S)]


class ValueElement(Element):
    """Element of one state and the equation a subclass gives."""

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.b = b


class FixedPointElement(ValueElement):
    """Fixed point of the cosine."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the fixed point equation."""
        return [EquationLaw(s.x == cos(s.x), target=s.x)]


class NoSolutionElement(ValueElement):
    """Equation without a real solution."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation."""
        return [EquationLaw(s.x**2 == -1, target=s.x)]


class MeanFieldElement(ValueElement):
    """Member whose equation reads the total of its group."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the coupled equation of one member."""
        return [EquationLaw(s.x == s.b + total(s.x) / 10, target=s.x)]


class SoftmaxFieldElement(ValueElement):
    """Member whose equation reads the log-sum-exp of its group."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the coupled equation of one member."""
        return [EquationLaw(s.x == s.b - log_sum_exp(s.x), target=s.x)]


class TermElement(ValueElement):
    """Element whose equation an interaction adds to."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the first term of the equation."""
        return [EquationLaw(s.x, 3.0, target=s.x)]


class StrayTargetElement(ValueElement):
    """Element with an equation whose target is not a state."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation."""
        return [EquationLaw(s.x == 1, target=var('z'))]


class NoTargetElement(ValueElement):
    """Element with an equation without a target."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation."""
        return [EquationLaw(s.x == 1)]


class RateElement(ValueElement):
    """Element with a rate law."""

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the rate."""
        return [RateLaw(s.x, 1.0)]


class CyclicElement(Element):
    """Two states that give each other."""

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the two equations."""
        return [EquationLaw(s.x == s.y / 2 + 1, target=s.x), EquationLaw(s.y == s.x / 2 + 1, target=s.y)]


class CournotElement(Element):
    """Best responses of a duopoly with cross holdings."""

    def __init__(
        self,
        label: str,
        *,
        q_i: Annotated[float, StateSpec()] = 0.0,
        q_j: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 10.0,
        c_i: Annotated[float, ParameterSpec()] = 1.0,
        c_j: Annotated[float, ParameterSpec()] = 2.0,
        lambda_i: Annotated[float, ParameterSpec()] = 0.0,
        lambda_j: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.q_i = q_i
        self.q_j = q_j
        self.alpha = alpha
        self.c_i = c_i
        self.c_j = c_j
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the conditions of Forrest (2018, equation 21.5)."""
        return [
            EquationLaw(s.alpha - 2 * s.q_i - s.q_j == s.c_i / (1 + s.lambda_i), target=s.q_i),
            EquationLaw(s.alpha - 2 * s.q_j - s.q_i == s.c_j / (1 + s.lambda_j), target=s.q_j),
        ]


class TermInteraction(Interaction):
    """Interaction that adds a term to the equation of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gain: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gain = gain

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the term of the equation."""
        return [EquationLaw(0, s.gain, target=s.target.x)]


def _simulated(system: System, **settings: object) -> Simulation:
    """Return the simulation of a static system after its run."""
    return Simulation(system, **settings).check().prepare().run(1.0)


def _issues(system: System, code: str) -> list[IssueSpec]:
    """Return the issues of a system with a code."""
    return [item for item in PassCompiler().check().prepare(system).report_.issues if item.code == code]


def test_run_survival_at_dose() -> None:
    """Test that a closed form under static time gives one frame with its value."""
    system = System(elements=[SurvivalElement('a')], time=StaticTime())
    frames = _simulated(system).trajectory_.values('a.S')
    assert compile_system(system).form == FormKind.STATIC
    assert frames.shape == (1,)
    assert abs(frames[0] - np.exp(-0.72)) < 1e-12


def test_run_without_equations() -> None:
    """Test that a static system without laws runs its observables in one frame on its initial states."""
    group = GroupElement('g', element=ValueElement, size=4, values={'x': np.arange(1.0, 5.0)})
    trajectory = _simulated(
        System(elements=[group], time=StaticTime()),
        observables=[ObservableSpec('all', group='g', variable='x', reduce='sum')],
    ).trajectory_
    assert trajectory.values('g.x').tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert trajectory.values('all').tolist() == [10.0]


def test_compile_system_exact_linear_solution() -> None:
    """Test that linear best responses compile to the symbolic solution of Forrest (2018, equation 21.6)."""
    problem = compile_system(System(elements=[CournotElement('a')], time=StaticTime()))
    names = ('alpha', 'c_i', 'c_j', 'lambda_i', 'lambda_j')
    alpha, c_i, c_j, lambda_i, lambda_j = (problem.symbols[f'a.{name}'] for name in names)
    expected = (alpha - 2 * c_i / (1 + lambda_i) + c_j / (1 + lambda_j)) / 3
    assert (SR(problem.equations['a.q_i']) - expected).simplify_full().is_trivial_zero()
    assert problem.unknowns.size == 0


def test_run_cyclic_equations() -> None:
    """Test that two equations that give each other solve jointly."""
    trajectory = _simulated(System(elements=[CyclicElement('a')], time=StaticTime())).trajectory_
    assert abs(trajectory.values('a.x')[0] - 2) < 1e-12
    assert abs(trajectory.values('a.y')[0] - 2) < 1e-12


def test_run_adds_equations_of_one_target() -> None:
    """Test that the equations of one target from an element and an interaction add."""
    system = System(
        elements=[TermElement('a')],
        interactions=[TermInteraction('i', target='a')],
        time=StaticTime(),
    )
    assert abs(_simulated(system).trajectory_.values('a.x')[0] - 5.0) < 1e-12


def test_run_numerical_solution() -> None:
    """Test that an equation without an explicit solution solves numerically."""
    system = System(elements=[FixedPointElement('a')], time=StaticTime())
    assert compile_system(system).unknowns.tolist() == [0]
    assert abs(_simulated(system).trajectory_.values('a.x')[0] - 0.7390851332151607) < 1e-9


def test_run_group_coupled_through_total() -> None:
    """Test that a group target gives one unknown per member, coupled through a reduction."""
    group = GroupElement('g', element=MeanFieldElement, size=5, values={'b': np.arange(1.0, 6.0)})
    frames = _simulated(System(elements=[group], time=StaticTime())).trajectory_.values('g.x')
    assert frames[0].tolist() == pytest.approx([4.0, 5.0, 6.0, 7.0, 8.0], abs=1e-8)


def test_run_group_coupled_through_log_sum_exp() -> None:
    """Test that the Jacobian of a log-sum-exp solves members that read it, from a guess far from the solution."""
    values = {'b': np.arange(1.0, 6.0), 'x': np.full(5, 10.0)}
    group = GroupElement('g', element=SoftmaxFieldElement, size=5, values=values)
    x = _simulated(System(elements=[group], time=StaticTime())).trajectory_.values('g.x')[0]
    assert x == pytest.approx(np.arange(1.0, 6.0) - logsumexp(x), abs=1e-8)


def test_run_no_solution() -> None:
    """Test that a solve that does not converge names the equation and the residual."""
    simulation = Simulation(System(elements=[NoSolutionElement('a')], time=StaticTime())).check().prepare()
    with pytest.raises(SolverError, match=r'x.*largest residual'):
        simulation.run(1.0)


def test_prepare_compiler_target_not_a_state() -> None:
    """Test that a target that is not a state is reported with the law."""
    found = _issues(System(elements=[StrayTargetElement('a')], time=StaticTime()), 'equation-target')
    assert len(found) == 1
    assert 'z' in found[0].message
    assert found[0].path.startswith('a')


def test_prepare_compiler_without_target() -> None:
    """Test that an equation without a target under static time is reported."""
    assert len(_issues(System(elements=[NoTargetElement('a')], time=StaticTime()), 'equation-target')) == 1


def test_prepare_compiler_rate_law_under_static_time() -> None:
    """Test that a rate law under static time keeps its time issue."""
    found = _issues(System(elements=[RateElement('a')], time=StaticTime()), 'law-time')
    assert len(found) == 1
    assert 'RateLaw' in found[0].message
