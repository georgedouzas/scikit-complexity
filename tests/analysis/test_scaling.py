"""Test finite-size scaling collapse on the Ising and percolation transitions."""

import math

import numpy as np
import pytest

from skcomplex.analysis import BinderAnalysis, ClusterAnalysis, scaling_collapse
from skcomplex.core.errors import AnalysisError

from .conftest import ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))
PERCOLATION = 0.5927


def _spanning(grids: list[np.ndarray]) -> float:
    """Return the share of grids with a cluster joining the opposite edges of their first axis."""
    return ClusterAnalysis('spanning').check().prepare(grids).run().probability_


def _susceptibilities(magnetizations: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    """Return the susceptibility of every size at every temperature."""
    return {
        side: BinderAnalysis('sweep', parameters=ISING_TEMPERATURES, size=side**2, temperatures=ISING_TEMPERATURES)
        .check()
        .prepare(np.column_stack(samples))
        .run()
        .susceptibility_
        for side, samples in magnetizations.items()
    }


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
            [_spanning([rng.random((side, side)) < p for _ in range(600)]) for p in probabilities],
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
