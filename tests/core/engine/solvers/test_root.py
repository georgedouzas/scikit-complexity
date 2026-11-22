"""Test the static form and the solver of its equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, exp, var
from scipy.special import logsumexp

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, ConstraintLaw
from skcomplex.core.components.laws.utils import log_sum_exp, total
from skcomplex.core.engine.compile import IssueSpec
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

VALUES = {'x': 1.0, 'b': 1.0}
COURNOT = {'q_i': 0.0, 'q_j': 0.0, 'alpha': 10.0, 'c_i': 1.0, 'c_j': 2.0, 'lambda_i': 0.0, 'lambda_j': 0.0}


def _survival(s):
    """Return the survival at a dose."""
    a = s.target
    return [ConstraintLaw(exp(-a.alpha * a.D - a.beta * a.D**2) == a.S, target=a.S)]


def _fixed_point(s):
    """Return the fixed point of the cosine."""
    x = s.target.x
    return [ConstraintLaw(x == cos(x), target=x)]


def _no_solution(s):
    """Return an equation without a real solution."""
    x = s.target.x
    return [ConstraintLaw(x**2 == -1, target=x)]


def _mean_field(s):
    """Return the equation of one member, which reads the total of its group."""
    x = s.target.x
    return [ConstraintLaw(x == s.target.b + total(x) / 10, target=x)]


def _softmax_field(s):
    """Return the equation of one member, which reads the log-sum-exp of its group."""
    x = s.target.x
    return [ConstraintLaw(x == s.target.b - log_sum_exp(x), target=x)]


def _term(s):
    """Return the first term of the equation, which an interaction adds to."""
    return [ConstraintLaw(s.target.x, 3.0, target=s.target.x)]


def _stray_target(s):
    """Return an equation whose target is not a state."""
    return [ConstraintLaw(s.target.x == 1, target=var('z'))]


def _no_target(s):
    """Return an equation without a target."""
    return [ConstraintLaw(s.target.x == 1)]


def _cyclic(s):
    """Return two equations that give each other."""
    x, y = s.target.x, s.target.y
    return [ConstraintLaw(x == y / 2 + 1, target=x), ConstraintLaw(y == x / 2 + 1, target=y)]


def _cournot(s):
    """Return the best responses of a duopoly with cross holdings, the conditions of Forrest (2018, equation 21.5)."""
    a = s.target
    return [
        ConstraintLaw(a.alpha - 2 * a.q_i - a.q_j == a.c_i / (1 + a.lambda_i), target=a.q_i),
        ConstraintLaw(a.alpha - 2 * a.q_j - a.q_i == a.c_j / (1 + a.lambda_j), target=a.q_j),
    ]


def _static(laws, values=None, size=None, interactions=()):
    """Return the element `a`, or a group `g` of a size, whose interaction `laws` sets laws of the namespace."""
    label = 'a' if size is None else 'g'
    element = Element(label, values=values or VALUES, size=size)
    own = Interaction('laws', roles={'target': label}, laws=laws)
    return System(elements=[element], interactions=[own, *interactions])


class TermInteraction(Interaction):
    """Interaction that adds a term to the equation of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gain: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gain = gain

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the term of the equation."""
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [ConstraintLaw(0, context.gain, target=context.target.x)]


def _simulated(system: System, **settings: object) -> Trajectory:
    """Return the trajectory of a static system."""
    return system.prepare().simulate(until=1.0, **settings)


def _issues(system: System, code: str) -> list[IssueSpec]:
    """Return the issues of a system with a code, read from the composite the compiler checks."""
    return [item for item in PassCompiler().prepare(system._get_composite()).report_.issues if item.code == code]


def test_run_survival_at_dose() -> None:
    """Test that a closed form under static time gives one frame with its value."""
    system = _static(_survival, {'S': 0.0, 'alpha': 0.3, 'beta': 0.03, 'D': 2.0})
    frames = _simulated(system).get_values('a.S')
    assert system.prepare()._problem.form == FormKind.STATIC
    assert frames.shape == (1,)
    assert abs(frames[0] - np.exp(-0.72)) < 1e-12


def test_run_without_equations() -> None:
    """Test that a static system without laws runs its observables in one frame on its initial states."""
    group = Element('g', values={'x': np.arange(1.0, 5.0), 'b': np.ones(4)}, size=4)
    trajectory = _simulated(
        System(elements=[group]),
        observables=[Observable('all', group='g', variable='x', reduce='sum')],
    )
    assert trajectory.get_values('g.x').tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert trajectory.get_values('all').tolist() == [10.0]


def test_compiler_exact_linear_solution() -> None:
    """Test that linear best responses compile to the symbolic solution of Forrest (2018, equation 21.6)."""
    problem = _static(_cournot, COURNOT).prepare()._problem
    names = ('alpha', 'c_i', 'c_j', 'lambda_i', 'lambda_j')
    alpha, c_i, c_j, lambda_i, lambda_j = (problem.symbols[f'a.{name}'] for name in names)
    expected = (alpha - 2 * c_i / (1 + lambda_i) + c_j / (1 + lambda_j)) / 3
    assert (SR(problem.equations['a.q_i']) - expected).simplify_full().is_trivial_zero()
    assert problem.unknowns.size == 0


def test_run_cyclic_equations() -> None:
    """Test that two equations that give each other solve jointly."""
    trajectory = _simulated(_static(_cyclic, {'x': 0.0, 'y': 0.0}))
    assert abs(trajectory.get_values('a.x')[0] - 2) < 1e-12
    assert abs(trajectory.get_values('a.y')[0] - 2) < 1e-12


def test_run_adds_equations_of_one_target() -> None:
    """Test that the equations of one target from an element and an interaction add."""
    system = _static(_term, interactions=[TermInteraction('i', target='a')])
    assert abs(_simulated(system).get_values('a.x')[0] - 5.0) < 1e-12


def test_run_numerical_solution() -> None:
    """Test that an equation without an explicit solution solves numerically."""
    system = _static(_fixed_point)
    assert system.prepare()._problem.unknowns.tolist() == [0]
    assert abs(_simulated(system).get_values('a.x')[0] - 0.7390851332151607) < 1e-9


def test_run_group_coupled_through_total() -> None:
    """Test that a group target gives one unknown per member, coupled through a reduction."""
    system = _static(_mean_field, {'x': np.ones(5), 'b': np.arange(1.0, 6.0)}, size=5)
    frames = _simulated(system).get_values('g.x')
    assert frames[0].tolist() == pytest.approx([4.0, 5.0, 6.0, 7.0, 8.0], abs=1e-8)


def test_run_group_coupled_through_log_sum_exp() -> None:
    """Test that the Jacobian of a log-sum-exp solves members that read it, from a guess far from the solution."""
    values = {'b': np.arange(1.0, 6.0), 'x': np.full(5, 10.0)}
    x = _simulated(_static(_softmax_field, values, size=5)).get_values('g.x')[0]
    assert x == pytest.approx(np.arange(1.0, 6.0) - logsumexp(x), abs=1e-8)


def test_run_no_solution() -> None:
    """Test that a solve that does not converge names the equation and the residual."""
    with pytest.raises(SolverError, match=r'x.*largest residual'):
        _simulated(_static(_no_solution))


def test_prepare_compiler_target_not_a_state() -> None:
    """Test that a target that is not a state is reported with the law."""
    found = _issues(_static(_stray_target), 'equation-target')
    assert len(found) == 1
    assert 'z' in found[0].message
    assert found[0].path.startswith('laws')


def test_prepare_compiler_without_target() -> None:
    """Test that an equation without a target under static time is reported."""
    assert len(_issues(_static(_no_target), 'equation-target')) == 1
