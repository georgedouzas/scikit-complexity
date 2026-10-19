"""Test component types defined from declarations instead of classes."""

import pickle  # nosec B403: the tests pickle and load only objects they create

import numpy as np
from sklearn.base import clone

from skcomplex.core.base import Domain, VariableKind, declarations_of, parameter, role, state
from skcomplex.core.model import Element, Interaction, Rate, System
from skcomplex.core.runtime import Simulation

Stock = Element.define('Stock', states={'n': state(1.0, domain=Domain.NONNEGATIVE)}, parameters={'k': 0.0})
Growth = Interaction.define(
    'Growth',
    roles=['target'],
    parameters={'r': parameter(0.5, domain=Domain.POSITIVE)},
    laws=lambda s: [Rate(s.target.n, s.r * s.target.n)],
)


def _decay_laws(self, s):
    """Return the decay of the stock at its own rate `k`."""
    return [Rate(s.n, -s.k * s.n)]


Decaying = Element.define('Decaying', states={'n': 1.0}, parameters={'k': 0.2}, laws=_decay_laws)


class StockClass(Element):
    """Stock written as a class."""

    n: float = state(1.0, domain=Domain.NONNEGATIVE)
    k: float = parameter(0.0)


class GrowthClass(Interaction):
    """Growth written as a class."""

    target: str = role()
    r: float = parameter(0.5, domain=Domain.POSITIVE)

    def laws(self, s):
        """Return exponential growth."""
        return [Rate(s.target.n, s.r * s.target.n)]


def test_defined_types_match_classes():
    """Test that a defined model simulates exactly like the same model written with classes."""
    defined = System(elements=[Stock('a', n=2.0)], interactions=[Growth('g', target='a', r=0.3)])
    written = System(elements=[StockClass('a', n=2.0)], interactions=[GrowthClass('g', target='a', r=0.3)])
    first = Simulation(defined, dt=0.5).run(until=2).trajectory_.states
    second = Simulation(written, dt=0.5).run(until=2).trajectory_.states
    assert np.array_equal(first, second)


def test_defined_types_declare_fields():
    """Test that defined types declare their states, parameters, and roles like classes."""
    assert set(declarations_of(Stock, VariableKind.STATE)) == {'n'}
    assert declarations_of(Stock, VariableKind.STATE)['n'].domain is Domain.NONNEGATIVE
    assert set(declarations_of(Stock, VariableKind.PARAMETER)) == {'k'}
    assert set(declarations_of(Growth, VariableKind.ROLE)) == {'target'}
    assert issubclass(Stock, Element)
    assert issubclass(Growth, Interaction)


def test_defined_types_form_groups():
    """Test that a group of 100 members of a defined type compiles once and exposes its parameters."""
    system = System(
        elements=[Stock.group('herd', 100, n=np.linspace(1, 2, 100))],
        interactions=[Growth('g', target='herd')],
    )
    trajectory = Simulation(system, dt=1).run(until=1).trajectory_
    assert trajectory['herd.n'].shape == (2, 100)
    assert np.allclose(trajectory['herd.n'][-1], np.linspace(1, 2, 100) * np.exp(0.5), rtol=1e-3)
    assert Growth('g', target='herd', r=0.7).get_params() == {'label': 'g', 'r': 0.7, 'target': 'herd'}


def test_defined_types_clone_and_pickle():
    """Test that instances of defined types clone and pickle with their values."""
    growth = Growth('g', target='a', r=0.7)
    assert clone(growth).get_params() == growth.get_params()
    data = pickle.dumps(System(elements=[Stock('a', n=3.0)], interactions=[growth]))
    restored = pickle.loads(data)  # noqa: S301  # nosec B301: loads data the test just dumped
    assert restored.elements[0].n == 3.0
    assert restored.interactions[0].r == 0.7


def test_laws_function_receives_component():
    """Test that a laws function with two arguments receives the component and the namespace."""
    trajectory = Simulation(System(elements=[Decaying('a', n=1.0)]), dt=1).run(until=1).trajectory_
    assert np.isclose(trajectory['a.n'][-1], np.exp(-0.2), rtol=1e-3)
