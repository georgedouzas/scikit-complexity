"""Test the `_neutral` module of `skcomplex.systems.biology.ecology`."""

import pytest

from skcomplex.systems.biology.ecology import (
    neutral_macroecology,
)

from ..conftest import _run_until


def test_neutral_abundances_settle_at_mean():
    """Test that neutral birth, death, and immigration settle at the mean abundance `b0 / (d1 - b1)`."""
    trajectory = _run_until(neutral_macroecology(50), 60, dt=10.0, random_state=0)
    assert trajectory.values('species.n')[-3:].mean() == pytest.approx(10, rel=0.2)
