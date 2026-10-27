"""Test the `_runner` module of `skcomplex.experiments`."""

import os
import subprocess  # nosec B404: the test runs a script it writes itself with this interpreter
import sys

import numpy as np
import pytest

from skcomplex.core.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.system import System
from skcomplex.experiments import (
    Experiment,
    GridDesign,
)
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


def last_value(trajectory):
    """Return the last infected count."""
    return float(trajectory.values('I.n')[-1])


SCRIPT_WITHOUT_GUARD = """
import skcomplex.models.mathematics.dynamical_systems
from skcomplex.analysis import orbit_diagram
from typing import Annotated
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.laws.utils import series

orbit_diagram('logistic_map', 'map.r', [2.9, 3.2], 'map.x', transient=10, keep=2, dt=1, executor='process')
"""


def test_factory_and_system_give_same_run():
    """Test that factory arguments and dotted paths of a built system give the same run."""
    by_argument = Experiment(logistic_growth, GridDesign(r=[0.7]), until=10, outputs=['population.n']).run()
    by_path = Experiment(logistic_growth(), GridDesign(**{'growth.r': [0.7]}), until=10, outputs=['population.n']).run()
    assert np.array_equal(by_argument.series('population.n'), by_path.series('population.n'))


@pytest.mark.slow
def test_ensemble_is_reproducible_across_executors():
    """Test that 20 stochastic runs with seed 7 give identical results serially and on four processes."""
    design = GridDesign(stochastic=[True])
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
    result = Experiment(
        sir,
        GridDesign(stochastic=[True]),
        replicates=20,
        until=50,
        outputs=['I.n'],
        random_state=7,
    ).run()
    assert len({tuple(series) for series in result.series('I.n')}) == 20


def test_orbit_diagram_sweep():
    """Test that a sweep of 200 growth rates of the logistic map keeps the last 100 values of each run."""
    system = System(elements=[LogisticMapElement('map')])
    rates = np.linspace(2.8, 4.0, 200)
    result = Experiment(system, GridDesign(**{'map.r': rates}), until=400, warmup=301, outputs=['map.x']).run()
    assert len(result.rows) == 200
    assert result.column('map.r').tolist() == rates.tolist()
    assert result.series('map.x').shape == (200, 100)


def test_summary_across_replicates():
    """Test that the summary gives the mean, standard deviation, and quantiles per design row and scenario."""
    result = Experiment(
        sir,
        GridDesign(stochastic=[True], gamma=[0.1, 0.2]),
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
        Experiment('logistic_growth', GridDesign(**{'growth.rate': [1.0]}), until=1).run()


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
