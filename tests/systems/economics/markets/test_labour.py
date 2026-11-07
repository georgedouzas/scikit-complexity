"""Test the `_labour` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import describe
from skcomplex.systems.economics.markets import LabourMarketSystem


def test_labour_market_flow_balance():
    """Unemployment settles at s / (s + f) and every hire takes an offer of the same step."""
    separation = 0.05
    trajectory = (
        Simulation(
            built := LabourMarketSystem(150, 30, separation=separation).prepare().system_,
            random_state=0,
            observables=describe('labour_market').observables_of(built),
        )
        .prepare()
        .run(until=150)
        .trajectory_
    )
    window = slice(50, None)
    finding = np.nansum(trajectory.values('labour.volume')[window]) / np.nansum(
        trajectory.values('searching')[window],
    )
    unemployment = trajectory.values('unemployment')[window].mean()
    assert 0.05 < finding < 0.95
    assert abs(unemployment - separation / (separation + finding)) < 0.03
    workers = trajectory.agents('workers', 150)
    hired = workers.values['hired'] >= 0
    assert hired.sum() > 50
    np.testing.assert_array_equal(workers.values['hired'][hired], workers.values['offer'][hired])
