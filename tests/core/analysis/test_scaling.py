"""Test Binder cumulants, their crossings, and finite-size scaling collapse on the Ising and percolation transitions."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import (
    binder_crossings,
    binder_cumulant,
    order_parameter_sweep,
    scaling_collapse,
    spanning_probability,
)
from skcomplex.core.base import AnalysisError

from .conftest import ISING_SIZES, ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))
PERCOLATION = 0.5927


def _susceptibilities(magnetizations: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    """Return the susceptibility of every size at every temperature."""
    return {
        side: order_parameter_sweep(
            ISING_TEMPERATURES,
            samples,
            side**2,
            temperatures=ISING_TEMPERATURES,
        ).susceptibility
        for side, samples in magnetizations.items()
    }


def test_binder_crossings_ising_critical_point(ising_magnetizations):
    """Test that the Binder cumulants of Ising lattices of sides 16, 32, and 64 cross within 1 percent of T_c."""
    cumulants = {side: [binder_cumulant(samples) for samples in ising_magnetizations[side]] for side in ISING_SIZES}
    crossings = binder_crossings(ISING_TEMPERATURES, cumulants)
    np.testing.assert_allclose(crossings, CRITICAL, rtol=0.01)
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        binder_crossings(ISING_TEMPERATURES, {16: cumulants[16]})


def test_scaling_collapse_same_on_two_transitions(ising_magnetizations):
    """Test that the Ising susceptibility and the spanning probability collapse better than with wrong exponents."""
    susceptibilities = _susceptibilities(ising_magnetizations)
    exact = scaling_collapse(ISING_TEMPERATURES, susceptibilities, critical=CRITICAL, nu=1.0, exponent=7 / 4)
    wrong = scaling_collapse(ISING_TEMPERATURES, susceptibilities, critical=CRITICAL, nu=1.2, exponent=7 / 4 * 1.2)
    fitted = scaling_collapse(
        ISING_TEMPERATURES,
        susceptibilities,
        critical=CRITICAL,
        nu=1.3,
        exponent=7 / 4,
        fit=('nu',),
    )
    assert exact.quality < wrong.quality
    assert fitted.nu == pytest.approx(1, rel=0.1)
    rng = np.random.default_rng(0)
    probabilities = np.linspace(0.56, 0.62, 13)
    spanning = {
        side: np.asarray(
            [spanning_probability([rng.random((side, side)) < p for _ in range(600)]) for p in probabilities],
        )
        for side in (16, 32, 64)
    }
    exact = scaling_collapse(probabilities, spanning, critical=PERCOLATION, nu=4 / 3)
    wrong = scaling_collapse(probabilities, spanning, critical=PERCOLATION, nu=4 / 3 * 1.2)
    fitted = scaling_collapse(probabilities, spanning, critical=PERCOLATION, nu=1.0, fit=('nu',))
    assert exact.quality < wrong.quality
    assert fitted.nu == pytest.approx(4 / 3, rel=0.1)
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        scaling_collapse(probabilities, {32: spanning[32]}, critical=PERCOLATION, nu=1.0)
