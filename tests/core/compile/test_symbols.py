"""Test that components receive Sage symbols, and that laws reject SymPy expressions."""

import pytest
import sympy  # noqa: TID251  # the test passes a SymPy expression to check that laws reject it

from skcomplex.core.base import Domain, Expression, SymbolicVector, ValidationError, domain_of, role
from skcomplex.core.compile import compile_system
from skcomplex.core.model import EuclideanSpace, Interaction, Particle, Rate, System
from skcomplex.models.biology.ecology import Population

RECEIVED: dict[str, object] = {}


class Recorder(Interaction):
    """Interaction that records the symbols it receives and adds no change."""

    target: str = role()

    def laws(self, s):
        """Record the symbols and return a zero rate on the velocity."""
        RECEIVED.update(x=s.target.x, m=s.target.m)
        return [Rate(s.target.v, 0)]


class SympyGrowth(Interaction):
    """Interaction whose rate is a SymPy expression."""

    target: str = role()

    def laws(self, s):
        """Return a SymPy rate."""
        return [Rate(s.target.n, sympy.Symbol('r') * 2)]


def test_components_receive_sage_symbols_and_vectors():
    """Test that scalar symbols are Sage expressions with their domains and vector states are Sage vectors."""
    system = System(elements=[Particle('p')], interactions=[Recorder('record', target='p')], space=EuclideanSpace(3))
    compile_system(system)
    assert isinstance(RECEIVED['x'], SymbolicVector)
    assert len(RECEIVED['x']) == 3
    assert all(isinstance(component, Expression) and component.is_symbol() for component in RECEIVED['x'])
    assert RECEIVED['m'].is_real()
    assert domain_of(RECEIVED['m']) is Domain.POSITIVE


def test_sympy_rate_is_rejected():
    """Test that a SymPy rate raises a validation error that names the law, the interaction, and Sage."""
    system = System(elements=[Population('herd')], interactions=[SympyGrowth('growth', target='herd')])
    with pytest.raises(ValidationError, match=r'Rate law of growth has the SymPy value .*Sage'):
        compile_system(system)
