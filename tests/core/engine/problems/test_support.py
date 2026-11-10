"""Test the support declarations of solvers and backends."""

import inspect

import pytest

from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime, LatticeSpace, StaticTime
from skcomplex.core.engine import solvers
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import (
    BaseSolver,
    FractionalABMSolver,
    GrunwaldLetnikovSolver,
    L1Solver,
    Solver,
)
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.specs import SupportSpec
from skcomplex.core.types import FormKind, OrderKind

SOLVERS = [
    item
    for item in vars(solvers).values()
    if inspect.isclass(item)
    and issubclass(item, BaseSolver)
    and not inspect.isabstract(item)
    and item not in {BaseSolver, Solver}
]


def test_empty_support_accepts_any_class():
    """Test that an empty field supports any class, and `None` in the space field supports no space."""
    assert SupportSpec().get_missing('time', [StaticTime]) == []
    assert SupportSpec(space={None}).get_missing('space', [None, LatticeSpace]) == ['LatticeSpace']


def test_support_accepts_subclasses():
    """Test that a class is supported when it is a subclass of a supported class."""
    assert SupportSpec(time={ContinuousTime}).get_missing('time', [ContinuousTime, DiscreteTime]) == ['DiscreteTime']


def test_default_ode_solver_lacks_noise():
    """Test that the default ODE solver supports rates but not noise."""
    assert DEFAULT_SOLVERS[FormKind.ODE]._get_support().get_missing('laws', [RateLaw, NoiseLaw]) == ['NoiseLaw']


@pytest.mark.parametrize('solver', SOLVERS, ids=lambda solver: solver.__name__)
def test_every_solver_declares_orders(solver):
    """Test that every solver declares the orders it integrates."""
    assert solver._get_support().orders


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_backends_support_every_order(backend):
    """Test that every backend supports integer and fractional orders."""
    assert backend._get_support().orders == set(OrderKind)


@pytest.mark.parametrize('solver', [GrunwaldLetnikovSolver, L1Solver, FractionalABMSolver])
def test_fractional_solvers_support_every_order(solver):
    """Test that the fractional solvers integrate integer and fractional orders."""
    assert solver._get_support().orders == set(OrderKind)


def test_ode_solver_supports_integer_orders():
    """Test that the default ODE solver integrates only integer orders."""
    assert DEFAULT_SOLVERS[FormKind.ODE]._get_support().orders == {OrderKind.INTEGER}
