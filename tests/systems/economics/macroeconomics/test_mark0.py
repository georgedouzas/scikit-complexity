"""Test the `_mark0` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np

from skcomplex.systems.economics.macroeconomics import Mark0System

from ..conftest import _run


def test_mark0_settles_at_full_employment_without_a_shock():
    """Test that the baseline of Table 1 is in the full employment phase, with few bankruptcies."""
    trajectory = _run(Mark0System(), 300)
    assert trajectory.get_values('economy.unemployment')[-50:].mean() < 0.05
    assert trajectory.get_values('economy.bankruptcies')[-100:].sum() < 0.05 * 500 * 100


def test_mark0_strong_shock_tips_into_a_recession_that_easy_credit_prevents():
    """Test the L shape of a strong long demand shock without policy and the recovery with adaptive credit."""
    shock = {'shock_length': 9, 'dc': 0.5}
    recession = _run(Mark0System(**shock), 250).get_values('economy.unemployment')
    assert recession[200:].mean() > 0.5
    helped = _run(Mark0System(**shock, credit='adaptive'), 250)
    assert helped.get_values('economy.unemployment')[200:].mean() < 0.1
    assert np.sum(helped.get_values('economy.bankruptcies')[100:110]) == 0
