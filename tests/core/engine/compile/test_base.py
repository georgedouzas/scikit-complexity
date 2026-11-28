"""Test the `base` module."""

import inspect
from types import SimpleNamespace
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine import compile as package
from skcomplex.core.engine.compile import BaseCompiler, BasePass, OperatorCompiler
from skcomplex.core.engine.compile._base import _ValidationReport
from skcomplex.core.engine.compile._compilation import _Issue
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import RoleElementError, ValidationError, ValidationWarning
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.engineering.control import PidInvertedPendulumSystem


def test_base_compiler_resolve_backend():
    """Test `BaseCompiler._resolve_backend` on a small case."""
    assert type(BaseCompiler._resolve_backend(None, 10)).__name__ == 'NumpyBackend'


def test_base_pass_get_replaced():
    """Test `BasePass._get_replaced` on a small case."""
    x = var('x')
    assert str(BasePass._get_replaced(RateLaw(x, -x), value=-2 * x).value) == '-2*x'


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_bases_and_the_operator_compiler_are_public():
    """The bases, the operator compiler, the flattening pass, and values are public, and the other passes private.

    A compiler of a system package flattens the tree of a system with the flattening pass.
    """
    expected = {'BaseCompiler', 'BasePass', 'BaseLowering', 'FlattenPass', 'HilbertSpaceError', 'OperatorCompiler'}
    assert expected <= set(package.__all__)
    assert all(name.endswith('Spec') for name in set(package.__all__) - expected)
    for compiler in (PassCompiler, OperatorCompiler):
        assert issubclass(compiler, package.BaseCompiler)
    assert not hasattr(package, PassCompiler.__name__)
    for name in ('LayoutPass', 'ProblemPass', 'Compilation', 'ObservableEvaluator'):
        assert not hasattr(package, name)


def test_base_compiler_prepares_on_a_system():
    """Test that a compiler compiles a `System` and a ready-made system through their composites."""
    pendulum = PidInvertedPendulumSystem(label='pendulum')
    for system in (pendulum, System(systems=[pendulum])):
        assert PassCompiler().prepare(system).run().problem_.form == FormKind.ODE


def test_compiler_run_raises_every_error_of_its_report():
    """Test that running a compiler raises every error of its report with its path and code, and emits its warnings."""
    compiler = PassCompiler().prepare(PidInvertedPendulumSystem(label='pendulum'))
    compiler.report_ = _ValidationReport(
        (
            _Issue('role', 'growth', 'The role is unbound.', 'Bind it.'),
            _Issue('law-domain', 'drift', 'The law targets a categorical state.', 'Use a jump law.'),
            _Issue('note', 'system', 'Something to know.', 'Nothing to do.', severity='warning'),
        ),
    )
    with pytest.warns(ValidationWarning, match='Something to know'), pytest.raises(ValidationError) as error:
        compiler.run()
    assert '2 error(s)' in str(error.value)
    assert 'growth: The role is unbound. Bind it. [role]' in str(error.value)
    assert '[law-domain]' in str(error.value)


class CountingInteraction(Interaction):
    """Growth of a count, whose role accepts only integer values."""

    def __init__(self, label: str, *, target: Annotated[Role, 'integer']) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the growth of the count."""
        super()._prepare(context)
        self.laws_ = [RateLaw(context.target.n.diff(context.t), 1)]


@pytest.mark.parametrize(('value', 'valid'), [(3, True), (0.5, False)])
def test_compiler_checks_the_domain_a_role_accepts_on_plain_elements(value, valid):
    """Test that a role annotated `Annotated[Role, 'integer']` takes an integer count and refuses a real one."""
    system = System(
        elements=[Element('tally', values={'n': value})],
        interactions=[CountingInteraction('grow', target='tally')],
    )
    if valid:
        assert system.prepare().states_ == ('tally.n',)
        return
    with pytest.raises(RoleElementError, match="accepts the domains integer"):
        system.prepare()
