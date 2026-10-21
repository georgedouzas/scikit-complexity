"""Test experiments: sources, seeds, designs, scenarios, results, sensitivity, and calibration."""

import math
import os
import subprocess  # nosec B404: the test runs a script it writes itself with this interpreter
import sys

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.base import ValidationError, state
from skcomplex.core.experiments import (
    Experiment,
    Grid,
    LatinHypercube,
    RandomDesign,
    Rows,
    Scenario,
    SobolDesign,
    calibrate,
    sensitivity,
)
from skcomplex.core.model import Element, Observable, System, Update
from skcomplex.models.biology.ecology import logistic_growth
from skcomplex.models.biology.epidemiology import sir
from skcomplex.models.economics.resources import harvesting

LogisticMap = Element.define(
    'LogisticMap',
    states={'x': state(default=0.2)},
    parameters={'r': 3.0},
    laws=lambda s: [Update(s.x, s.r * s.x * (1 - s.x))],
)
Point = Element.define('Point', parameters={'x1': 0.0, 'x2': 0.0, 'x3': 0.0})


def last_value(trajectory):
    """Return the last infected count."""
    return float(trajectory['I.n'][-1])


def test_factory_and_system_give_same_run():
    """Test that factory arguments and dotted paths of a built system give the same run."""
    by_argument = Experiment(logistic_growth, Grid(r=[0.7]), until=10, outputs=['population.n']).run()
    by_path = Experiment(logistic_growth(), Grid(**{'growth.r': [0.7]}), until=10, outputs=['population.n']).run()
    assert np.array_equal(by_argument.series('population.n'), by_path.series('population.n'))


def test_ensemble_is_reproducible_across_executors():
    """Test that 20 stochastic runs with seed 7 give identical results serially and on four processes."""
    design = Grid(stochastic=[True])
    serial = Experiment(sir, design, replicates=20, until=50, outputs=['I.n'], random_state=7).run()
    parallel = Experiment(
        sir,
        design,
        replicates=20,
        until=50,
        outputs=['I.n'],
        random_state=7,
        executor='process',
    ).run()
    assert np.array_equal(serial.series('I.n'), parallel.series('I.n'))


def test_replicates_are_independent():
    """Test that no two of 20 stochastic replicates have identical trajectories."""
    result = Experiment(sir, Grid(stochastic=[True]), replicates=20, until=50, outputs=['I.n'], random_state=7).run()
    assert len({tuple(series) for series in result.series('I.n')}) == 20


def test_orbit_diagram_sweep():
    """Test that a sweep of 200 growth rates of the logistic map keeps the last 100 values of each run."""
    system = System(elements=[LogisticMap('map')])
    rates = np.linspace(2.8, 4.0, 200)
    result = Experiment(system, Grid(**{'map.r': rates}), until=400, warmup=301, outputs=['map.x']).run()
    assert len(result.rows) == 200
    assert result.column('map.r').tolist() == rates.tolist()
    assert result.series('map.x').shape == (200, 100)


@pytest.mark.parametrize(
    'design',
    [
        RandomDesign(16, r=(0.2, 1.0), random_state=0),
        LatinHypercube(16, r=(0.2, 1.0), random_state=0),
        SobolDesign(16, r=(0.2, 1.0), random_state=0),
    ],
    ids=['random', 'latin', 'sobol'],
)
def test_sampled_designs_stay_in_bounds(design):
    """Test that sampled designs give the requested number of rows within the bounds."""
    values = [row['r'] for row in design.rows()]
    assert len(values) == 16
    assert all(0.2 <= value <= 1.0 for value in values)


def test_designs_cross_and_concatenate():
    """Test that designs cross with `*` and concatenate with `+`."""
    crossed = Grid(r=[1, 2]) * Grid(K=[10, 20, 30])
    joined = Grid(r=[1, 2]) + Rows([{'r': 3}])
    assert len(crossed.rows()) == 6
    assert [row['r'] for row in joined.rows()] == [1, 2, 3]


def test_policy_changes_at_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    result = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0).run()
    baseline, quota = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed


def test_summary_across_replicates():
    """Test that the summary gives the mean, standard deviation, and quantiles per design row and scenario."""
    result = Experiment(
        sir,
        Grid(stochastic=[True], gamma=[0.1, 0.2]),
        replicates=5,
        until=30,
        outputs=[last_value],
        random_state=1,
    ).run()
    summary = result.summary()
    assert len(summary) == 2
    for row in summary:
        assert row['runs'] == 5
        assert {'last_value.mean', 'last_value.std', 'last_value.q0.05', 'last_value.q0.5', 'last_value.q0.95'} <= set(
            row,
        )
    assert result.to_frame().shape[0] == 10


def test_unknown_path_is_rejected():
    """Test that an override of a path that is neither a state nor a parameter names the available paths."""
    with pytest.raises(ValidationError, match=r'growth\.rate is neither a state nor a parameter'):
        Experiment('logistic_growth', Grid(**{'growth.rate': [1.0]}), until=1).run()


def _ishigami(s):
    """Return the Ishigami function of the point's coordinates."""
    return sin(s.p.x1) + 7 * sin(s.p.x2) ** 2 + 0.1 * s.p.x3**4 * sin(s.p.x1)


def test_sobol_indices_of_ishigami():
    """Test that Sobol first-order indices of the Ishigami function match the known values within 0.05."""
    system = System(
        elements=[Point('p')],
        observables=[Observable('y', _ishigami)],
    )
    bounds = {f'p.x{index}': (-math.pi, math.pi) for index in (1, 2, 3)}
    result = sensitivity(system, bounds, 'y', method='sobol', n=2048, until=0, random_state=0)
    expected = {'p.x1': 0.3139, 'p.x2': 0.4424, 'p.x3': 0.0}
    for name, value in expected.items():
        assert result.indices[name]['S1'] == pytest.approx(value, abs=0.05)
    assert result.ranking()[0] == 'p.x1'
    assert result.indices['p.x1']['ST'] == pytest.approx(0.5576, abs=0.05)


def test_morris_and_one_at_a_time_rank_parameters():
    """Test that Morris screening and one-at-a-time effects rank the carrying capacity first for the final size."""
    bounds = {'growth.r': (0.5, 1.0), 'growth.K': (50.0, 150.0)}
    morris = sensitivity('logistic_growth', bounds, 'population.n', method='morris', n=10, until=20, random_state=0)
    oat = sensitivity('logistic_growth', bounds, 'population.n', method='oat', until=20)
    assert morris.ranking()[0] == 'growth.K'
    assert oat.ranking()[0] == 'growth.K'


def test_calibration_recovers_logistic_parameters():
    """Test that calibration recovers r = 0.8 and K = 500 from data with 1% noise and fits with R² above 0.99."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    data = {'population.n': (times, truth * (1 + 0.01 * rng.standard_normal(times.size)))}
    result = calibrate('logistic_growth', data, {'growth.r': (0.1, 2.0), 'growth.K': (100.0, 1000.0)})
    assert result.estimates['growth.r'] == pytest.approx(0.8, rel=0.05)
    assert result.estimates['growth.K'] == pytest.approx(500, rel=0.05)
    assert result.r2 > 0.99
    assert result.covariance is not None


SCRIPT_WITHOUT_GUARD = """
import skcomplex.models.mathematics.dynamical_systems
from skcomplex.core.analysis import orbit_diagram

orbit_diagram('logistic_map', 'map.r', [2.9, 3.2], 'map.x', transient=10, keep=2, dt=1, executor='process')
"""


def test_process_executor_names_the_missing_main_guard(tmp_path):
    """Test that a script without a main guard that runs an experiment on processes gets a named error saying so."""
    script = tmp_path / 'script.py'
    script.write_text(SCRIPT_WITHOUT_GUARD)
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(item for item in sys.path if item)}
    completed = subprocess.run(  # noqa: S603  # nosec B603: runs the test's own script with this interpreter
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=300,
        check=False,
    )
    assert completed.returncode != 0
    assert 'ExperimentError' in completed.stderr
    assert "if __name__ == '__main__':" in completed.stderr
    assert "executor='thread'" in completed.stderr
