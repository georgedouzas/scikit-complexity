"""Test the `_flux_balance` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems import create_simulation
from skcomplex.systems.biology.common.metabolism import FluxVariabilityAnalysis
from skcomplex.systems.biology.systems_biology import flux_balance


def test_flux_balance_reaches_the_growth_of_raman():
    """Test that the default toy network grows at 50, as Raman Eq. 8.9 gives."""
    trajectory = create_simulation('flux_balance').prepare().run().trajectory_
    assert float(trajectory.values('cell.v_vbio')[0]) == pytest.approx(50, abs=1e-9)
    assert float(trajectory.values('cell.v_v6')[0]) == pytest.approx(1000)


def test_flux_balance_variability_gives_the_ranges_of_raman():
    """Test the ranges of Raman Eq. 8.12 at the optimum of the toy network."""
    ranges = FluxVariabilityAnalysis().prepare(flux_balance()).run().ranges_
    expected = {
        'v1': (0, 600),
        'v2': (0, 600),
        'v3': (0, 600),
        'v4': (550, 550),
        'v5': (50, 50),
        'v6': (1000, 1000),
        'v7': (100, 100),
        'b1': (600, 600),
        'b2': (1000, 1000),
    }
    for name, values in expected.items():
        assert ranges.loc[name].tolist() == pytest.approx(values, abs=1e-6), name


def test_flux_balance_parsimonious_uses_the_smallest_total():
    """Test that the parsimonious solution reaches the same growth with the smallest total flux."""
    trajectory = create_simulation('flux_balance', parsimonious=True).prepare().run().trajectory_
    assert float(trajectory.values('cell.objective_0')[0]) == pytest.approx(50)
    assert float(trajectory.values('cell.objective_1')[0]) == pytest.approx(3950)


def test_flux_balance_oxygen_sweep_by_an_experiment():
    """Test that growth rises from 0.2117 without oxygen and stays at 0.8739 above the uptake of about 21.8."""
    levels = [0.0, 5.0, 10.0, 20.0, 25.0]
    experiment = Experiment(
        flux_balance,
        GridDesign(),
        {'oxygen': levels, 'network': ['ecoli']},
        until=0.0,
        outputs=['cell.objective_0'],
    )
    rows = experiment.prepare().run().result_.rows
    growth = [float(np.ravel(row['cell.objective_0'])[-1]) for row in rows]
    assert growth[0] == pytest.approx(0.2117, abs=1e-4)
    assert np.all(np.diff(growth) > 0)
    assert growth[-1] == pytest.approx(0.8739, abs=1e-4)
