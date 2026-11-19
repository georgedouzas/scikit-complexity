"""Test the `_layout` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, RateLaw, TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import count, total, where
from skcomplex.core.components.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

from ..conftest import QuadraticElement, phase_system, quadratic_system


class CountElement(Element):
    """Amount that cannot be negative beside a count of whole events."""

    def __init__(
        self,
        label: str,
        *,
        amount: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        events: Annotated[float, NN] = 0,
        level: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.amount = amount
        self.events = events
        self.level = level


def _counts(**values):
    """Return a count element whose three values stay constant."""
    constant = Interaction(
        'constant',
        roles={'target': 'a'},
        laws=lambda s: [
            UpdateLaw(s.target.amount, s.target.amount),
            UpdateLaw(s.target.events, s.target.events),
            UpdateLaw(s.target.level, s.target.level),
        ],
    )
    return System(elements=[CountElement('a', **values)], interactions=[constant])


def test_complex_state_reads_back_its_initial_value():
    """Test that a complex default is stored as a pair of floats and read back as a complex number."""
    problem = phase_system(z=1 + 2j).prepare()._problem
    assert problem.initial_state.tolist() == [1.0, 2.0]
    assert problem.states.read(problem.initial_state, 'osc.z') == 1 + 2j


def test_real_state_rejects_complex_value():
    """Test that a complex value of a real state names the state, its domain, and the complex domain."""
    with pytest.raises(ValidationError, match=r'osc\.omega = 1j.*its domain Real Field.*domain=CC'):
        phase_system(omega=1j).prepare()


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ({'amount': -1.0}, r'a\.amount = -1\.0 is not in its domain \[0, \+oo\)\.'),
        ({'events': 1.5}, r'a\.events = 1\.5 is not in its domain Non negative integer semiring\.'),
    ],
)
def test_real_value_outside_real_domain_is_reported_outside_it(values, message):
    """Test that a real value outside a real domain is reported as outside it, with no word of an imaginary part."""
    with pytest.raises(ValidationError, match=message) as error:
        _counts(**values).prepare()
    assert 'imaginary' not in str(error.value)


def test_complex_value_of_real_state_names_its_imaginary_part():
    """Test that a complex value of a real state is reported as having a nonzero imaginary part."""
    with pytest.raises(ValidationError, match=r'a\.level = \(1\+2j\) has a nonzero imaginary part'):
        _counts(level=1 + 2j).prepare()


def test_real_state_accepts_complex_value_with_zero_imaginary_part():
    """Test that a complex value with a zero imaginary part is a real value."""
    problem = phase_system(omega=3 + 0j).prepare()._problem
    assert problem.parameters.read(problem.initial_parameters, 'osc.omega') == 3.0


def test_complex_parameter_per_site_is_read_and_updated():
    """Test that every site reads its own complex parameter, and that an update changes the next step."""
    c = np.arange(16).reshape(4, 4) * (1 + 1j)
    system = quadratic_system(QuadraticElement('grid', c=c, space=LatticeSpace((4, 4))))
    trajectory = system.prepare().simulate(until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], c)
    system.set_params(grid__c=np.full((4, 4), 2j))
    trajectory = system.prepare().simulate(until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], 2j)


class SpreadInteraction(Interaction):
    """Rule of the system that spreads a value of a group through a square matrix of weights."""

    def __init__(self, label: str, *, target: Role = 'nodes', weights=None) -> None:
        super().__init__(label)
        self.target = target
        self.weights = weights

    def _prepare(self, context=None):
        """Set the spread as a rule of the system."""
        super()._prepare(context)
        self.laws_ = [UpdateLaw(context, self.spread)]

    def spread(self, system):
        """Set every value to the weighted sum of the values."""
        nodes = system.get_agents('target')
        values = np.asarray(self.weights) @ np.array([node.x for node in nodes])
        for node, value in zip(nodes, values, strict=True):
            node.x = float(value)


def test_a_square_matrix_of_an_interaction_is_laid_out_as_its_entries():
    """Test that a square matrix setting of an interaction, such as weights, is laid out as its entries."""
    nodes = Element('nodes', values={'x': [1.0, 2.0]}, size=2)
    spread = SpreadInteraction('spread', weights=[[0.0, 1.0], [1.0, 0.0]])
    trajectory = System(elements=[nodes], interactions=[spread]).prepare().simulate(until=1, random_state=0)
    assert trajectory.get_agents('nodes', 1).values['x'].tolist() == [2.0, 1.0]


class Coupling(Interaction):
    """Coupling `x_i <- sum_j a_ij x_j` of the members of a group through a matrix."""

    def __init__(self, label: str, *, target: Role = 'nodes', a: object = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context):
        """Set the coupled update of every member."""
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [UpdateLaw(x, total(x, weights=context.a))]


@pytest.mark.parametrize('size', [3, 4])
def test_matrix_argument_shaped_by_the_members_of_a_group_is_a_matrix_parameter(size):
    """Test that an array argument of one row and one column per member couples the members, also for three."""
    matrix = np.arange(size * size, dtype=float).reshape(size, size) / 10
    nodes = Element('nodes', values={'x': np.arange(1.0, size + 1)}, size=size)
    system = System(elements=[nodes], interactions=[Coupling('coupling', a=matrix)]).prepare()
    trajectory = system.simulate(until=1)
    np.testing.assert_allclose(trajectory.get_values('nodes.x')[-1], matrix @ np.arange(1.0, size + 1))
    np.testing.assert_allclose(trajectory.get_values('coupling.a')[-1], matrix)


def test_a_law_counts_the_members_of_a_group_with_no_values():
    """Test that new nodes arrive at a rate of one until a law that counts the nodes of a bare group sees five."""
    space = NetworkSpace(nx.empty_graph(3, create_using=nx.DiGraph))
    request = TopologyRequestSpec('add-node', independent=True)
    growth = Interaction(
        'growth',
        roles={'target': 'nodes'},
        laws=lambda s: [JumpLaw(where(count(over=s.target) < 5, 1.0, 0), {request: 1})],
    )
    system = System(elements=[Element('nodes', space=space)], interactions=[growth], space=space).prepare()
    assert len(system.simulate(until=50, random_state=0).get_network(-1).nodes) == 5


def test_a_law_counts_the_members_of_a_group_with_no_values_off_a_network():
    """Test that a rate that counts the members of a bare group with no space reads their number."""
    counting = Interaction(
        'counting',
        roles={'target': 'stock', 'source': 'cells'},
        laws=lambda s: [RateLaw(s.target.x, count(over=s.source))],
    )
    system = System(elements=[Element('stock', values={'x': 0.0}), Element('cells', size=4)], interactions=[counting])
    assert system.prepare().simulate(until=1).get_values('stock.x')[-1] == pytest.approx(4.0)
