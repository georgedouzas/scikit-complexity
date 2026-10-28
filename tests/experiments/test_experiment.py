"""Test the `_experiment` module of `skcomplex.experiments`."""

import os
import subprocess  # nosec B404: the test runs a script it writes itself with this interpreter
import sys
from typing import Annotated

import numpy as np
import pytest

from skcomplex.analysis import BaseAnalysis
from skcomplex.core.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.experiments import Experiment, GridDesign, RowsDesign
from skcomplex.models.biology.ecology import logistic_growth
from skcomplex.models.biology.epidemiology import sir


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


class OrnsteinUhlenbeckElement(Element):
    """Ornstein-Uhlenbeck process that relaxes to zero with white noise."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, x: Annotated[object, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the drift and the noise of the process."""
        return [RateLaw(s.x, -s.x), NoiseLaw(s.x, 0.5)]


class PeakAnalysis(BaseAnalysis):
    """Height and time of the peak of one series."""

    def __init__(self, variable='I.n'):
        self.variable = variable

    def _get_data_kind(self):
        return 'series'

    def _prepare(self, source):
        self.series_ = self._read_series(source, [self.variable])
        return self

    def _run(self):
        values = self.series_.values[:, 0]
        self.peak_ = float(values.max())
        self.peak_time_ = float(self.series_.times[values.argmax()])
        return self


def last_value(trajectory):
    """Return the last infected count."""
    return float(trajectory.values('I.n')[-1])


SCRIPT_WITHOUT_GUARD = """
from skcomplex.experiments import Experiment
from skcomplex.models.biology.epidemiology import sir

Experiment(sir, replicates=2, until=5, executor='process').prepare().run()
"""


def test_factory_and_system_give_same_run():
    """Test that factory arguments and dotted paths of a built system give the same run."""
    by_argument = Experiment(logistic_growth, GridDesign(), {'r': [0.7]}, until=10, outputs=['population.n'])
    by_path = Experiment(logistic_growth(), GridDesign(), {'growth.r': [0.7]}, until=10, outputs=['population.n'])
    first, second = by_argument.prepare().run().result_, by_path.prepare().run().result_
    assert np.array_equal(first.series('population.n'), second.series('population.n'))


def test_analyses_of_stochastic_sir_replicates():
    """Test that 20 stochastic SIR replicates with a peak analysis give one row per run with the peak of that run."""
    experiment = (
        Experiment(
            sir,
            RowsDesign([{'stochastic': True}]),
            replicates=20,
            analyses=[PeakAnalysis()],
            until=50,
            outputs=['I.n'],
            random_state=7,
        )
        .prepare()
        .run()
    )
    rows = experiment.result_.rows
    assert len(rows) == 20
    for row, series in zip(rows, experiment.result_.series('I.n'), strict=True):
        assert row['peakanalysis.peak'] == series.max()
        assert 0 <= row['peakanalysis.peak_time'] <= 50
    assert len({row['peakanalysis.peak'] for row in rows}) > 1
    assert experiment.analyses[0].__dict__ == {'variable': 'I.n'}


def test_analyses_of_one_class_are_numbered():
    """Test that two analyses of one class get numbered column prefixes."""
    experiment = Experiment('sir', analyses=[PeakAnalysis('I.n'), PeakAnalysis('S.n')], until=10).prepare().run()
    row = experiment.result_.rows[0]
    assert row['peakanalysis-1.peak'] > 0
    assert row['peakanalysis-2.peak'] == experiment.result_.series('S.n')[0].max()


@pytest.mark.slow
def test_serial_and_process_runs_are_identical():
    """Test that 20 stochastic runs with seed 7 and a peak analysis give identical results serially and on processes."""
    results = [
        Experiment(
            sir,
            RowsDesign([{'stochastic': True}]),
            replicates=20,
            analyses=[PeakAnalysis()],
            until=50,
            outputs=['I.n'],
            random_state=7,
            executor=executor,
        )
        .prepare()
        .run()
        .result_
        for executor in ('serial', 'process')
    ]
    assert np.array_equal(results[0].series('I.n'), results[1].series('I.n'))
    assert results[0].rows == results[1].rows


def test_ensemble_of_ornstein_uhlenbeck_replicates():
    """Test that the ensemble of 200 Ornstein-Uhlenbeck replicates has one row per replicate carrying its seed."""
    system = System(elements=[OrnsteinUhlenbeckElement('ou')])
    experiment = Experiment(system, replicates=200, until=2, dt=0.1, outputs=['ou.x'], random_state=0)
    times, values, seeds = experiment.prepare().run().result_.ensemble('ou.x')
    assert values.shape == (200, times.size)
    assert seeds.tolist() == list(range(200))
    assert len({tuple(row) for row in values}) == 200
    assert values[:, -1].mean() == pytest.approx(np.exp(-2.0), abs=0.1)


def test_replicates_are_independent():
    """Test that no two of 20 stochastic replicates have identical trajectories."""
    experiment = Experiment(
        sir,
        RowsDesign([{'stochastic': True}]),
        replicates=20,
        until=50,
        outputs=['I.n'],
        random_state=7,
    )
    assert len({tuple(series) for series in experiment.prepare().run().result_.series('I.n')}) == 20


def test_orbit_diagram_sweep():
    """Test that a sweep of 200 growth rates of the logistic map keeps the last 100 values of each run."""
    system = System(elements=[LogisticMapElement('map')])
    rates = np.linspace(2.8, 4.0, 200)
    experiment = Experiment(system, GridDesign(), {'map.r': rates}, until=400, warmup=301, outputs=['map.x'])
    result = experiment.prepare().run().result_
    assert len(result.rows) == 200
    assert result.column('map.r').tolist() == rates.tolist()
    assert result.series('map.x').shape == (200, 100)


def test_summary_across_replicates():
    """Test that the summary gives the mean, standard deviation, and quantiles per design row and scenario."""
    experiment = Experiment(
        sir,
        GridDesign(),
        {'stochastic': [True], 'gamma': [0.1, 0.2]},
        replicates=5,
        until=30,
        outputs=[last_value],
        random_state=1,
    )
    result = experiment.prepare().run().result_
    summary = result.summary()
    assert len(summary) == 2
    for row in summary:
        assert row['runs'] == 5
        assert {'last_value.mean', 'last_value.std', 'last_value.q0.05', 'last_value.q0.5', 'last_value.q0.95'} <= set(
            row,
        )
    assert result.to_frame().shape[0] == 10


def test_zero_replicates_are_rejected():
    """Test that an experiment of zero replicates builds and fails its check with an error naming the replicates."""
    with pytest.raises(ValidationError, match='replicates = 0'):
        Experiment('sir', replicates=0).check()


def test_bounds_without_design_are_rejected():
    """Test that bounds without a design fail the check."""
    with pytest.raises(ValidationError, match='need a design'):
        Experiment('sir', bounds={'gamma': [0.1]}).check()


def test_unknown_path_is_rejected():
    """Test that an override of a path that is neither a state nor a parameter names the available paths."""
    experiment = Experiment('logistic_growth', GridDesign(), {'growth.rate': [1.0]}, until=1).prepare()
    with pytest.raises(ValidationError, match=r'growth\.rate is neither a state nor a parameter'):
        experiment.run()


@pytest.mark.slow
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
