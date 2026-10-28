"""Test the `_optimal_therapy` module of `skcomplex.models.biology.oncology`."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models import run
from skcomplex.models.biology.oncology import TumourImmuneChemotherapyElement


def test_optimal_chemotherapy_shrinks_the_tumour():
    """Test that the optimal infusion ends with a smaller tumour than no treatment, starting at the largest dose."""
    treated = run('optimal_chemotherapy').trajectory_
    untreated = (
        Simulation(System(elements=[TumourImmuneChemotherapyElement('patient')]), dt=0.25)
        .check()
        .run(until=30.0)
    )
    assert treated.values('patient.u')[0] == pytest.approx(1.0)
    assert treated.values('patient.A')[-1] < 0.5 * untreated.trajectory_.values('patient.A')[-1]


def test_optimal_immunotherapy_clears_the_tumour():
    """Test that the optimal adoptive therapy and IL-2 clear the tumour, which grows without them."""
    treated = run('optimal_immunotherapy').trajectory_
    assert treated.values('site.y')[-1] < 0.01
    assert treated.values('site.u1')[0] == pytest.approx(1.0)
    untreated = run('optimal_immunotherapy', w1=1e6, w2=1e6, settings={'until': 20.0}).trajectory_
    assert untreated.values('site.y')[-1] > 1.0
