"""Test the `_stock_flow` module of `skcomplex.systems.general.dynamics`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.analysis import loops
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.engine.runtime import ConservationObserver
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
        return {'drain': ('level', None, lambda s: s.target.k * s.target.level)}


def tub(inflow: float, outflow: float) -> StockFlowSystem:
    """Return a bathtub with a faucet and a drain."""
    return StockFlowSystem(
        stocks={'water': 50.0},
        flows={
            'faucet': (None, 'water', lambda s: s.target.inflow),
            'drain': ('water', None, lambda s: s.target.outflow),
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
        flows={'forth': ('a', 'b', lambda s: 0.3 * s.target.a), 'back': ('b', 'a', lambda s: 0.1 * s.target.b)},
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


def test_stock_flow_flows_with_the_environment_have_one_stock():
    """Test that a flow into the system and one out of it each change one stock, booked against the environment."""
    system = StockFlowSystem(
        stocks={'grain': 5.0},
        flows={'harvest': (None, 'grain', lambda s: 2.0), 'release': ('grain', None, lambda s: 2.0)},
    )
    trajectory = system.prepare().simulate(until=3.0, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('stocks.grain'), 5.0)
    np.testing.assert_allclose(trajectory.get_values('stocks.release'), 2.0)
    matrix = loops(trajectory, kind='flow-matrix')
    assert matrix.holders == ('environment', 'stocks.grain')
    np.testing.assert_allclose(matrix.matrix, [[-2.0, 2.0], [2.0, -2.0]])


def test_stock_flow_rejects_a_flow_of_no_stock():
    """Test that a flow that neither leaves nor enters a stock names the flow."""
    system = StockFlowSystem(stocks={'grain': 1.0}, flows={'pass': (None, None, lambda s: 1.0)})
    with pytest.raises(ValidationError, match=r"'pass' .* neither leaves nor enters"):
        system.prepare()


def test_stock_flow_rejects_a_negative_stock_it_allows_only_nonnegative():
    """Test that a stock allowed only nonnegative values rejects a negative one."""
    system = StockFlowSystem(
        stocks={'level': -1.0},
        flows={'drain': ('level', None, lambda s: s.target.level)},
        allowed={'level': RealSet.closed_open(0, oo)},
        element='tank',
    )
    with pytest.raises(ValidationError, match='level'):
        system.prepare()


def test_stock_flow_keeps_the_quantity_of_a_stock():
    """Test that the conservation monitor follows the quantity a stock holds."""
    system = StockFlowSystem(
        stocks={'water': 1.0},
        flows={'drain': ('water', None, lambda s: s.target.water)},
        quantities={'water': 'water'},
        element='tub',
    )
    monitor = ConservationObserver()
    system.prepare().simulate(until=1.0, dt=1.0, observers=[monitor])
    assert monitor.totals_['water'] == pytest.approx(np.exp(-1.0), rel=1e-4)
