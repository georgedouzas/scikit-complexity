"""Test the `_cell_survival` module of `skcomplex.systems.biology.oncology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems.biology.oncology import lq_cell_survival


def test_lq_cell_survival_fractionation() -> None:
    """Test that 30 fractions of 2 Gy at `alpha / beta = 10` give the biologically effective dose of 72 Gy."""
    system = lq_cell_survival(60.0, fractions=30.0, alpha=0.3, beta=0.03, delivery_time=1e-3)
    trajectory = Simulation(system).prepare().run(1.0).trajectory_
    assert len(trajectory.times) == 1
    assert trajectory.values('cells.effective_dose')[0] == pytest.approx(72.0, abs=1e-2)


def test_lq_cell_survival_protraction_tends_to_one() -> None:
    """Test that the factor of a constant dose rate tends to 1 as the delivery time tends to 0, and falls with it."""
    bounds = {'delivery_time': [10.0, 1.0, 0.1, 1e-3]}
    experiment = Experiment(lq_cell_survival, GridDesign(), bounds, until=1.0, outputs=['cells.protraction'])
    protraction = experiment.prepare().run().result_.series('cells.protraction')[:, 0].ravel()
    assert np.all(np.diff(protraction) > 0)
    assert protraction[-1] == pytest.approx(1.0, abs=1e-3)


def test_lq_cell_survival_curve_follows_the_closed_form() -> None:
    """Test that a sweep of the dose follows `exp(-alpha D - beta D^2)` for an acute dose."""
    doses = [0.0, 1.0, 2.0, 5.0, 10.0]
    experiment = Experiment(lq_cell_survival, GridDesign(), {'dose': doses}, until=1.0, outputs=['cells.survival'])
    survival = experiment.prepare().run().result_.series('cells.survival')[:, 0].ravel()
    expected = np.exp(-0.3 * np.array(doses) - 0.03 * np.array(doses) ** 2)
    np.testing.assert_allclose(survival, expected, rtol=1e-3)
