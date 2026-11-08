"""Test the `_optimal_therapy` module of `skcomplex.systems.biology.oncology`."""

import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.oncology import (
    OptimalChemotherapySystem,
    OptimalImmunotherapySystem,
    TumourImmuneChemotherapyElement,
)


def test_optimal_chemotherapy_shrinks_the_tumour():
    """Test that the optimal infusion ends with a smaller tumour than no treatment, starting at the largest dose."""
    treated = simulate(OptimalChemotherapySystem())
    untreated = simulate(System(elements=[TumourImmuneChemotherapyElement('patient')]), until=30.0, dt=0.25)
    assert treated.values('patient.u')[0] == pytest.approx(1.0)
    assert treated.values('patient.A')[-1] < 0.5 * untreated.values('patient.A')[-1]


def test_optimal_immunotherapy_clears_the_tumour():
    """Test that the optimal adoptive therapy and IL-2 clear the tumour, which grows without them."""
    treated = simulate(OptimalImmunotherapySystem())
    assert treated.values('site.y')[-1] < 0.01
    assert treated.values('site.u1')[0] == pytest.approx(1.0)
    untreated = simulate(OptimalImmunotherapySystem(w1=1e6, w2=1e6), until=20.0)
    assert untreated.values('site.y')[-1] > 1.0
