"""Test the support declarations of solvers and backends."""

import inspect

import pytest

from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.spaces import ContinuousTime, DiscreteTime, LatticeSpace, StaticTime
from skcomplex.engine import solvers
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.problems import Form, OrderKind, SupportSpec
from skcomplex.engine.solvers import (
    DEFAULT_SOLVERS,
    BaseSolver,
    FractionalABMSolver,
    GrunwaldLetnikovSolver,
    L1Solver,
)

SOLVERS = [
    item
    for item in vars(solvers).values()
    if inspect.isclass(item)
    and issubclass(item, BaseSolver)
    and not inspect.isabstract(item)
    and 'supports' in vars(item)
]


def test_empty_support_accepts_any_class():
    """Test that an empty field supports any class, and `None` in the space field supports no space."""
    assert SupportSpec().missing('time', [StaticTime]) == []
    assert SupportSpec(space={None}).missing('space', [None, LatticeSpace]) == ['LatticeSpace']


def test_support_accepts_subclasses():
    """Test that a class is supported when it is a subclass of a supported class."""
    assert SupportSpec(time={ContinuousTime}).missing('time', [ContinuousTime, DiscreteTime]) == ['DiscreteTime']


def test_default_ode_solver_lacks_noise():
    """Test that the default ODE solver supports rates but not noise."""
    assert DEFAULT_SOLVERS[Form.ODE].supports.missing('laws', [RateLaw, NoiseLaw]) == ['NoiseLaw']


@pytest.mark.parametrize('solver', SOLVERS, ids=lambda solver: solver.__name__)
def test_every_solver_declares_orders(solver):
    """Test that every solver declares the orders it integrates."""
    assert solver.supports.orders


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_backends_support_every_order(backend):
    """Test that every backend supports integer and fractional orders."""
    assert backend.supports.orders == set(OrderKind)


@pytest.mark.parametrize('solver', [GrunwaldLetnikovSolver, L1Solver, FractionalABMSolver])
def test_fractional_solvers_support_every_order(solver):
    """Test that the fractional solvers integrate integer and fractional orders."""
    assert solver.supports.orders == set(OrderKind)


def test_ode_solver_supports_integer_orders():
    """Test that the default ODE solver integrates only integer orders."""
    assert DEFAULT_SOLVERS[Form.ODE].supports.orders == {OrderKind.INTEGER}
