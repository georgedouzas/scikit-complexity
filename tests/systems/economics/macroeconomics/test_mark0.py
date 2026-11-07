"""Test the `_mark0` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np

from skcomplex.systems.economics.macroeconomics import mark0

from ..conftest import _run


def test_mark0_settles_at_full_employment_without_a_shock():
    """Test that the baseline of Table 1 is in the full employment phase, with few bankruptcies."""
    trajectory = _run(mark0(), 300)
    assert trajectory.values('economy.unemployment')[-50:].mean() < 0.05
    assert trajectory.values('economy.bankruptcies')[-100:].sum() < 0.05 * 500 * 100


def test_mark0_strong_shock_tips_into_a_recession_that_easy_credit_prevents():
    """Test the L shape of a strong long demand shock without policy and the recovery with adaptive credit."""
    shock = {'shock_length': 9, 'dc': 0.5}
    recession = _run(mark0(**shock), 250).values('economy.unemployment')
    assert recession[200:].mean() > 0.5
    helped = _run(mark0(**shock, credit='adaptive'), 250)
    assert helped.values('economy.unemployment')[200:].mean() < 0.1
    assert np.sum(helped.values('economy.bankruptcies')[100:110]) == 0
