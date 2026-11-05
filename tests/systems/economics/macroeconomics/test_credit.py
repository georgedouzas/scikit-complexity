"""Test the `_credit` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np

from skcomplex.systems import describe
from skcomplex.systems.economics.macroeconomics import credit_networks

from ..conftest import _run_observed


def test_credit_switching_concentrates_loans():
    """A higher intensity of switching concentrates loans in fewer banks and narrows the spread of rates."""
    slow = _run_observed(
        built := credit_networks(60, 8, intensity=0.5),
        40,
        observables=describe('credit_networks').observables_of(built),
    )
    fast = _run_observed(
        built := credit_networks(60, 8, intensity=20.0),
        40,
        observables=describe('credit_networks').observables_of(built),
    )
    assert fast.values('herfindahl')[-20:].mean() > slow.values('herfindahl')[-20:].mean()
    assert fast.values('credit.spread')[-20:].mean() < slow.values('credit.spread')[-20:].mean()


def test_credit_net_worth_moves_between_firms_and_banks():
    """Interest and bad debt move net worth between firms and banks, so the total changes by returns and entries."""
    trajectory = _run_observed(
        built := credit_networks(40, 5),
        20,
        observables=describe('credit_networks').observables_of(built),
    )
    total = trajectory.values('bank_equity') + trajectory.values('firm_worth')
    changes = trajectory.values('returns')[1:] + trajectory.values('injected')[1:]
    np.testing.assert_allclose(np.diff(total), changes, rtol=1e-6, atol=1e-6)
