"""Test the `_contractions` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, RateLaw
from skcomplex.core.components.laws.utils import contraction
from skcomplex.core.components.laws.utils._contractions import contractions_in
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import JaxBackend, NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import ParameterError, ValidationError
from skcomplex.core.specs import ParameterSpec, Role, RoleSpec
from skcomplex.core.systems import System

from .....conftest import NodeElement

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend, JaxBackend], ids=['numpy', 'numba', 'jax'])


class CubicInteraction(Interaction):
    """Rate of every member of a group, the contraction of a tensor of order 3 with the states of two members."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        T: object = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.T = T

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the contraction of the tensor with the states."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(context.t), contraction(context.T, x, x))]


def _cubic(tensor, x, target='species', backend='auto'):
    """Return a group whose rates contract a tensor with its states."""
    return System(
        'community',
        elements=[Element('species', values={'x': x}, size=len(x)), Element('single', values={'x': 0.0})],
        interactions=[CubicInteraction('cubic', target=target, T=tensor)],
        backend=backend,
    )


class ShapedCubicInteraction(Interaction):
    """Contraction of a tensor whose declared shape follows the members its role binds."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        T: Annotated[object, ParameterSpec(shape=('target', 'target', 'target'))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.T = T

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the contraction of the tensor with the states."""
        return [RateLaw(s.target.x, contraction(s.T, s.target.x, s.target.x))]


def _shaped_cubic(tensor, x, target='species'):
    """Return a group whose rates contract a tensor of a declared shape with its states."""
    return CompositeSystem(
        'community',
        elements=[GroupElement('species', element=NodeElement, size=len(x), values={'x': x}), NodeElement('single')],
        interactions=[ShapedCubicInteraction('cubic', target=target, T=tensor)],
    )


def test_contraction_keeps_its_tensor_and_expressions():
    """Test that a contraction keeps the tensor and the expressions as operands."""
    T, x = var('T x')
    assert [str(item) for item in contractions_in(x * contraction(T, x, 2 * x))[0].operands()] == ['T', 'x', '2*x']


def test_contraction_rejects_a_tensor_that_is_not_a_symbol():
    """Test that a tensor that is not a symbol raises an error."""
    with pytest.raises(ValidationError, match='symbol of a shaped parameter'):
        contraction(2.0, var('x'))


def test_contraction_rejects_no_expressions():
    """Test that a contraction without expressions raises an error."""
    with pytest.raises(ValidationError, match='one expression per index'):
        contraction(var('T'))


@BACKENDS
def test_contraction_equals_einsum(backend):
    """Test that a contraction of a random tensor of order 3 over 20 members equals `einsum('ijk,j,k->i', T, x, x)`."""
    rng = np.random.default_rng(0)
    tensor, x = rng.normal(size=(20, 20, 20)), rng.uniform(size=20)
    problem = _cubic(tensor, x, backend=backend()).prepare()._problem
    rate = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'species.x')
    np.testing.assert_allclose(rate, np.einsum('ijk,j,k->i', tensor, x, x), atol=1e-12)


@BACKENDS
def test_contraction_of_a_tensor_on_an_element(backend):
    """Test that a tensor whose role binds a single element reports the issue `parameter-shape`."""
    with pytest.raises(ParameterError, match='parameter-shape'):
        PassCompiler(backend=backend()).prepare(_shaped_cubic(np.zeros((1, 1, 1)), np.zeros(3), target='single')).run()
