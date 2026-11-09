"""Test the `_stock_flow` module of `skcomplex.systems.general.dynamics`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.laws import Sink, Source
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from skcomplex.systems.general.dynamics import StockFlowSystem
from skcomplex.systems.general.dynamics.types import FlowTable


class TankSystem(StockFlowSystem):
    """A tank drained at a rate proportional to its level, with a level that cannot fall below zero."""

    def __init__(self, level: float = 1.0, *, k: float = 0.5, label: str = 'system') -> None:
        super().__init__(label=label)
        self.level = level
        self.k = k

    def _get_element_label(self) -> str:
        return 'tank'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        return {
            'level': DeclaredVariableSpec(self.level, StateSpec(domain=RealSet.closed_open(0, oo))),
            'k': DeclaredVariableSpec(self.k, ParameterSpec()),
        }

    def _get_flows(self) -> FlowTable:
        return {'drain': ('level', None, lambda s: s.source.k * s.source.level)}


def tub(inflow: float, outflow: float) -> StockFlowSystem:
    """Return a bathtub with a faucet and a drain."""
    return StockFlowSystem(
        stocks={'water': 50.0},
        flows={
            'faucet': (None, 'water', lambda s: s.source.inflow),
            'drain': ('water', None, lambda s: s.source.outflow),
        },
        parameters={'inflow': inflow, 'outflow': outflow},
        element='tub',
    )


def test_stock_flow_bathtub_stays_level_and_reports_its_flows():
    """Test that equal faucet and drain keep the water constant and that the trajectory holds both flows."""
    trajectory = simulate(tub(5.0, 5.0), until=10.0, dt=1.0)
    np.testing.assert_allclose(trajectory.values('tub.water'), 50.0)
    assert {'tub.faucet', 'tub.drain'} <= set(trajectory.names)
    np.testing.assert_allclose(trajectory.values('tub.faucet'), 5.0)


def test_stock_flow_rejects_a_flow_of_an_unknown_stock():
    """Test that a flow of an undeclared stock names the flow and the stock."""
    system = StockFlowSystem(stocks={'water': 1.0}, flows={'leak': ('oil', None, lambda s: 1.0)})
    with pytest.raises(ValidationError, match=r"'leak'.*'oil'"):
        system.prepare()


def test_stock_flow_declared_domains_reject_a_negative_stock():
    """Test that a stock declared nonnegative rejects a negative value as an element with that domain does."""
    with pytest.raises(ValidationError) as rebased:
        simulate(TankSystem(level=-1.0), until=1.0)
    variables = {'level': DeclaredVariableSpec(-1.0, StateSpec(domain=RealSet.closed_open(0, oo)))}
    with pytest.raises(ValidationError) as element:
        simulate(CompositeSystem(elements=[DeclaredElement('tank', variables=variables)]), until=1.0)
    domain = next(line for line in str(element.value).splitlines() if 'tank.level = -1.0' in line)
    assert domain in str(rebased.value).splitlines()


def test_stock_flow_between_stocks_keeps_the_total():
    """Test that flows between stocks keep their total."""
    system = StockFlowSystem(
        stocks={'a': 10.0, 'b': 0.0},
        flows={'forth': ('a', 'b', lambda s: 0.3 * s.source.a), 'back': ('b', 'a', lambda s: 0.1 * s.source.b)},
    )
    trajectory = simulate(system, until=40.0, dt=1.0)
    np.testing.assert_allclose(trajectory.values('stocks.a') + trajectory.values('stocks.b'), 10.0, rtol=1e-9)
    assert trajectory.values('stocks.a')[-1] == pytest.approx(2.5, rel=1e-3)


def test_stock_flow_subclass_takes_its_own_parameters():
    """Test that a subclass lists its own parameters and drains its tank exponentially."""
    system = TankSystem()
    assert sorted(system.get_params(deep=False)) == ['k', 'label', 'level']
    trajectory = simulate(system, until=2.0, dt=1.0)
    assert trajectory.values('tank.level')[-1] == pytest.approx(np.exp(-1.0), rel=1e-5)
    assert 'tank.drain' in trajectory.names


def test_stock_flow_takes_a_labelled_source_and_sink():
    """Test that a labelled source and sink name the environment of a flow as `None` does unlabelled."""
    system = StockFlowSystem(
        stocks={'grain': 5.0},
        flows={
            'harvest': (Source('farms'), 'grain', lambda s: 2.0),
            'release': ('grain', Sink('consumers'), lambda s: 2.0),
        },
    )
    interactions = system.prepare().system_.interactions
    assert [interactions[0].state.label, interactions[1].into.label] == ['farms', 'consumers']
    trajectory = simulate(system, until=3.0, dt=1.0)
    np.testing.assert_allclose(trajectory.values('stocks.grain'), 5.0)


def test_stock_flow_rejects_a_flow_between_two_environments():
    """Test that a flow from a labelled source to a labelled sink names the flow."""
    system = StockFlowSystem(stocks={'grain': 1.0}, flows={'pass': (Source('farms'), Sink('market'), lambda s: 1.0)})
    with pytest.raises(ValidationError, match=r"'pass' .* neither leaves nor enters"):
        system.prepare()
