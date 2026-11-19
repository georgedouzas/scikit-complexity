"""Test the `_element` module of `skcomplex.core.components.elements`."""

from multiprocessing.reduction import ForkingPickler
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element, Quantity
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.spaces import (
    AxisSpec,
    ContinuousSpace,
    LatticeSpace,
    LayerSpec,
    NetworkSpace,
    PhaseSpace,
)
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import LabelError, ValidationError
from skcomplex.core.systems import System


class TallyElement(Element):
    """Element with a setting next to its value."""

    def __init__(self, label: str, *, value: float = 0.0, source: str = '') -> None:
        super().__init__(label)
        self.value = value
        self.source = source


class DecayingElement(Element):
    """Stock with its own decay rate."""

    def __init__(self, label: str, *, n: float = 1.0, k: float = 0.2) -> None:
        super().__init__(label)
        self.n = n
        self.k = k


class FirmElement(Element):
    """Firm with a debt, an equity, and the growth rate of its debt."""

    def __init__(self, label: str, *, debt: float = 1.0, equity: float = 2.0, r: float = 0.1) -> None:
        super().__init__(label)
        self.debt = debt
        self.equity = equity
        self.r = r


def _system(element, laws, **settings):
    """Return the system of one element under the laws of one interaction, a function of the namespace."""
    rule = Interaction('laws', roles={'target': element.label}, laws=laws)
    return System(elements=[element], interactions=[rule], **settings)


def _decay(s):
    """Return the decay of a stock at its own rate."""
    return [RateLaw(s.target.n, -s.target.k * s.target.n)]


def _debt(s):
    """Return the growth of the debt of a firm, slowed by the leverage."""
    return [RateLaw(s.target.debt, s.target.r * s.target.debt * (1 - _leverage(s.target)))]


def _leverage(s):
    """Return the leverage of a firm, its debt over its equity."""
    return s.debt / s.equity


def _firm_leverage(s):
    """Return the leverage of the firm of a system."""
    return _leverage(s.firm)


def test_element_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, and every value."""
    tally = TallyElement('size', source='farmers')
    assert tally.get_params() == {'label': 'size', 'source': 'farmers', 'value': 0.0}
    assert repr(tally) == "TallyElement(label='size', source='farmers')"


def test_element_rejects_unknown_arguments():
    """Test that an element cannot be made with an argument it does not declare."""
    with pytest.raises(TypeError, match="'foo'"):
        TallyElement('rabbits', foo=1)


def test_element_clones_and_pickles_with_its_values():
    """Test that an element clones and pickles with its values."""
    restored = ForkingPickler.loads(ForkingPickler.dumps(DecayingElement('a', n=3.0)))
    assert (restored.n, restored.k) == (3.0, 0.2)
    assert clone(DecayingElement('a', n=3.0)).n == 3.0


def test_validation_reports_an_element_with_an_invalid_label_by_its_path():
    """Test that validating a system reports a nested element whose label is not a string."""
    system = System(systems=[System('meadow', elements=[TallyElement(1)])])
    with pytest.raises(LabelError, match=r'^Label 1 in meadow/ is not valid'):
        PassCompiler().prepare(system)


def test_element_decays_at_its_own_rate():
    """Test that laws run with the values of the element they read."""
    trajectory = _system(DecayingElement('a', n=1.0), _decay).prepare().simulate(until=1, dt=1)
    assert np.isclose(trajectory.get_values('a.n')[-1], np.exp(-0.2), rtol=1e-3)


def test_expression_of_states_is_an_observable_of_the_laws():
    """Test that a law computes the leverage as an expression and the run records an observable of the same one."""
    system = _system(FirmElement('firm'), _debt)
    assert (
        str(PassCompiler().prepare(system).run().problem_.equations['firm.debt'])
        == '-debt_firm*r_firm*(debt_firm/equity_firm - 1)'
    )
    leverage = Observable('leverage', _firm_leverage)
    trajectory = system.prepare().simulate(until=2, dt=0.5, observables=[leverage])
    debt, equity = trajectory.get_values('firm.debt'), trajectory.get_values('firm.equity')
    np.testing.assert_allclose(trajectory.get_values('leverage'), debt / equity)


def test_agent_state_named_like_view_raises_error():
    """Test that an agent with a state named like an attribute of the agent view raises a label error."""

    def stay(agent, system):
        """Set the value the rule changes, so that it is a state."""
        agent.alive = agent.alive

    settlers = Element('settlers', values={'alive': 0.0}, size=2)
    with pytest.raises(LabelError, match=r'settlers declare the variables alive,.*reserved'):
        _system(settlers, lambda s: [UpdateLaw(s.target, stay)]).prepare()


def test_numpy_scalar_times_vector_is_rejected():
    """Test that a law value that NumPy built as an array raises a named error that names the state."""
    moving = Element('m', values={'v': [1.0, 2.0]})
    system = _system(
        moving,
        lambda s: [RateLaw(s.target.v, np.float64(2.0) * s.target.v)],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(ValidationError, match='NumPy array'):
        system.prepare()


def test_numpy_scalar_values_match_floats():
    """Test that NumPy scalar law values run like the same laws written with Python floats."""

    def numpy_laws(s):
        """Return the decay and the inflow written with NumPy scalars."""
        return [RateLaw(s.target.x, np.float64(-0.5) * s.target.x), RateLaw(s.target.x, np.float64(1.0))]

    def float_laws(s):
        """Return the decay and the inflow written with Python floats."""
        return [RateLaw(s.target.x, -0.5 * s.target.x), RateLaw(s.target.x, 1.0)]

    first, second = (
        _system(Element('d', values={'x': 2.0}), laws).prepare().simulate(until=2, dt=0.5).get_values('d.x')
        for laws in (numpy_laws, float_laws)
    )
    assert np.array_equal(first, second)


def test_jump_increment_symbol_is_rejected():
    """Test that a jump whose increment is a parameter raises a named error naming the state and the increment."""
    counter = Element('c', values={'n': 0.0, 'k': 2.0})
    with pytest.raises(ValidationError, match=r'jump of n_c has the increment k_c.*number'):
        _system(counter, lambda s: [JumpLaw(1.0, {s.target.n: s.target.k})]).prepare()


class _TableElement(Element):
    """Billiard table that wrongly stores its half side under the reserved name `size`."""

    def __init__(self, label: str, size: float = 1.5) -> None:
        super().__init__(label)
        self.size = size


@pytest.mark.parametrize(
    ('element', 'name'),
    [
        (_TableElement('table'), 'size'),
        (Element('gas', values={'x': 1.0}, size=0), 'size'),
        (Element('ball', values=[('x', 1.0)]), 'values'),
        (Element('plate', values={'u': 0.0}, space='lattice'), 'space'),
    ],
)
def test_prepare_rejects_a_reserved_name_used_for_a_value(element, name):
    """Test that a reserved name holding something else than its setting raises an error naming the reserved names."""
    with pytest.raises(ValidationError, match=rf"has {name} = .*label, size, space, values are reserved"):
        element.prepare()


def test_prepare_accepts_a_size_counted_by_numpy():
    """Test that a number of members computed with NumPy, such as the side of a grid squared, is a size."""
    side = np.int64(5)
    assert Element('namers', values={'words': 0.0}, size=side**2).prepare().size == 25


@pytest.mark.parametrize(
    ('on', 'space'),
    [('links', NetworkSpace(nx.path_graph(3))), ('edges', ContinuousSpace(n_dim=2)), ('edges', None)],
)
def test_prepare_rejects_an_on_that_names_no_links_of_a_network(on, space):
    """Test that `on` other than the edges, hyperedges, or a layer of a network raises an error with an example."""
    with pytest.raises(ValidationError, match=r"has on = .*'edges', 'hyperedges', or the name of a layer"):
        Element('links', values={'sign': 1}, space=space, on=on).prepare()


def test_prepare_accepts_an_on_that_names_a_layer_of_a_network():
    """Test that `on` may name a layer of a network space, whose edges hold one member each."""
    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('roads', nx.complete_graph(3))])
    assert Element('links', values={'sign': 1}, space=space, on='roads').prepare().on == 'roads'


@pytest.mark.skip(reason='engine gap: a value of its own named on is read as the part of a network space')
def test_prepare_keeps_a_state_of_its_own_named_on():
    """Test that an element that declares its own state `on`, such as a switch, is not read as the part of a space."""

    class SwitchElement(Element):
        """Switch whose state `on` is 1 when it is closed."""

        def __init__(self, label: str, on: float = 1.0) -> None:
            super().__init__(label)
            self.on = on

    assert SwitchElement('switch', on=0.0).prepare().on == 0.0


def test_element_on_a_space_has_the_tags_of_its_space():
    """Test that an element on a phase space asks for a solver of phase spaces, and one without a space for none."""
    space = PhaseSpace(LatticeSpace(4), [AxisSpec('v', 2, -1.0, 1.0)])
    assert Element('gas', values={'x': 1.0}, space=space).get_tags().space == frozenset({'phase space'})
    assert not Element('ball', values={'x': 1.0}).get_tags().space


class VoterElement(Element):
    """Plain element with an opinion of 0 or 1 and a real confidence."""

    def __init__(self, label: str, *, opinion: Annotated[int, {0, 1}] = 0, confidence: float = 0.5) -> None:
        super().__init__(label)
        self.opinion = opinion
        self.confidence = confidence


def test_plain_element_has_the_domains_of_all_its_values_before_prepare():
    """Test that a plain element, whose laws decide its states, has the domain tags of every numeric value."""
    assert VoterElement('voter').get_tags().domains == {'boolean', 'real'}
    assert Element('counter', values={'n': 3}).get_tags().domains == {'integer'}


class HeatedElement(Element):
    """Plain element with an annotated temperature and a plain mass."""

    def __init__(self, label: str, *, T: Annotated[float, Quantity('energy')] = 1.0, m: float = 2.0) -> None:
        super().__init__(label)
        self.T = T
        self.m = m


def test_element_gives_the_hints_of_its_value_arguments():
    """Test that the hints of an element are the annotations of its value arguments, in their order."""
    assert HeatedElement('body')._get_hints() == {'T': Annotated[float, Quantity('energy')], 'm': float}
    assert Element('ball', values={'x': 1.0})._get_hints() == {}
