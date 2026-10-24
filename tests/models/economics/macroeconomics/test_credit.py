"""Test the `_credit` module of `skcomplex.models.economics.macroeconomics`."""

import numpy as np

from skcomplex.models import describe
from skcomplex.models.economics.macroeconomics import credit_networks

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
    assert fast['herfindahl'][-20:].mean() > slow['herfindahl'][-20:].mean()
    assert fast['credit.spread'][-20:].mean() < slow['credit.spread'][-20:].mean()


def test_credit_net_worth_moves_between_firms_and_banks():
    """Interest and bad debt move net worth between firms and banks, so the total changes by returns and entries."""
    trajectory = _run_observed(
        built := credit_networks(40, 5),
        20,
        observables=describe('credit_networks').observables_of(built),
    )
    total = trajectory['bank_equity'] + trajectory['firm_worth']
    changes = trajectory['returns'][1:] + trajectory['injected'][1:]
    np.testing.assert_allclose(np.diff(total), changes, rtol=1e-6, atol=1e-6)
