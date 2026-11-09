"""Test the `_sweep` module of `skcomplex.core.experiments`."""

from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.experiments import sweep
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import SIRSystem


class LogisticMapElement(Element):
    """Logistic map element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        r: Annotated[object, ParameterSpec()] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r

    def _get_laws(self, s):
        """Return the laws of the logistic map element."""
        return [UpdateLaw(s.x, s.r * s.x * (1 - s.x))]


def last_value(trajectory):
    """Return the last infected count."""
    return float(trajectory.get_values('I.x')[-1])


def test_sweep_of_one_parameter_gives_one_row_per_value():
    """Test that a sweep over one model parameter gives one tidy row per value with its output."""
    result = sweep(LogisticGrowthSystem(), {'K': [50.0, 100.0]}, until=40, outputs=['population.x'])
    assert result.column('K').tolist() == [50.0, 100.0]
    np.testing.assert_allclose(result.column('population.x'), [50.0, 100.0], rtol=1e-3)
    assert list(result.to_frame().columns[:4]) == ['K', 'scenario', 'replicate', 'seed']


def test_orbit_diagram_sweep():
    """Test that a sweep of 200 growth rates of the logistic map keeps the last 100 values of each run."""
    system = CompositeSystem(elements=[LogisticMapElement('map')])
    rates = np.linspace(2.8, 4.0, 200)
    result = sweep(system, {'map.r': rates}, until=400, warmup=301, outputs=['map.x'])
    assert len(result.rows) == 200
    assert result.column('map.r').tolist() == rates.tolist()
    assert result.series('map.x').shape == (200, 100)


def test_sampled_sweep_stays_within_bounds():
    """Test that a sampled sweep runs `n` rows within the bounds."""
    result = sweep(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, design='sobol', n=4, until=1, random_state=0)
    assert len(result.rows) == 4
    assert all(0.5 <= value <= 1.0 for value in result.column('r'))


def test_summary_across_replicates():
    """Test that the summary gives the mean, standard deviation, and quantiles per row."""
    result = sweep(
        SIRSystem(stochastic=True),
        {'gamma': [0.1, 0.2]},
        replicates=5,
        until=30,
        outputs=[last_value],
        random_state=1,
    )
    summary = result.summary()
    assert len(summary) == 2
    for row in summary:
        assert row['runs'] == 5
        assert {'last_value.mean', 'last_value.std', 'last_value.q0.05', 'last_value.q0.5', 'last_value.q0.95'} <= set(
            row,
        )
    assert result.to_frame().shape[0] == 10
