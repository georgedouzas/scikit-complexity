"""Test the `_stock_flow` module of `skcomplex.systems.general.dynamics`."""

import numpy as np
import pytest

from skcomplex.core.components.laws import Sink, Source
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.general.dynamics import StockFlowSystem
from skcomplex.systems.general.dynamics.types import FlowTable


class TankSystem(StockFlowSystem):
    """A tank drained at a rate proportional to its level."""

    def __init__(
        self,
        level: float = 1.0,
        *,
        k: float = 0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.level = level
        self.k = k

    def _get_element_label(self) -> str:
        return 'tank'

    def _get_stocks(self) -> dict[str, object]:
        return {'level': self.level}

    def _get_parameters(self) -> dict[str, object]:
        return {'k': self.k}

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
    trajectory = tub(5.0, 5.0).prepare().simulate(until=10.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('tub.water'), 50.0)
    assert {'tub.faucet', 'tub.drain'} <= set(trajectory.names)
    np.testing.assert_allclose(trajectory.get_values('tub.faucet'), 5.0)


def test_stock_flow_rejects_a_flow_of_an_unknown_stock():
    """Test that a flow of an undeclared stock names the flow and the stock."""
    system = StockFlowSystem(stocks={'water': 1.0}, flows={'leak': ('oil', None, lambda s: 1.0)})
    with pytest.raises(ValidationError, match=r"'leak'.*'oil'"):
        system.prepare()


def test_stock_flow_between_stocks_keeps_the_total():
    """Test that flows between stocks keep their total."""
    system = StockFlowSystem(
        stocks={'a': 10.0, 'b': 0.0},
        flows={'forth': ('a', 'b', lambda s: 0.3 * s.source.a), 'back': ('b', 'a', lambda s: 0.1 * s.source.b)},
    )
    trajectory = system.prepare().simulate(until=40.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('stocks.a') + trajectory.get_values('stocks.b'), 10.0, rtol=1e-9)
    assert trajectory.get_values('stocks.a')[-1] == pytest.approx(2.5, rel=1e-3)


def test_stock_flow_subclass_takes_its_own_parameters():
    """Test that a subclass lists its own parameters and drains its tank exponentially."""
    system = TankSystem()
    assert sorted(system.get_params(deep=False)) == ['backend', 'k', 'label', 'level', 'solver']
    trajectory = system.prepare().simulate(until=2.0, dt=1.0)
    assert trajectory.get_values('tank.level')[-1] == pytest.approx(np.exp(-1.0), rel=1e-5)
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
    trajectory = system.prepare().simulate(until=3.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('stocks.grain'), 5.0)


def test_stock_flow_rejects_a_flow_between_two_environments():
    """Test that a flow from a labelled source to a labelled sink names the flow."""
    system = StockFlowSystem(stocks={'grain': 1.0}, flows={'pass': (Source('farms'), Sink('market'), lambda s: 1.0)})
    with pytest.raises(ValidationError, match=r"'pass' .* neither leaves nor enters"):
        system.prepare()
