"""Test the `_credit` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np

from skcomplex.systems.economics.macroeconomics import CreditNetworksSystem

from ..conftest import _run_observed


def test_credit_switching_concentrates_loans():
    """A higher intensity of switching concentrates loans in fewer banks and narrows the spread of rates."""
    slow = _run_observed(
        CreditNetworksSystem(60, 8, intensity=0.5).prepare().system_,
        40,
        observables=CreditNetworksSystem().get_run_settings()['observables'],
    )
    fast = _run_observed(
        CreditNetworksSystem(60, 8, intensity=20.0).prepare().system_,
        40,
        observables=CreditNetworksSystem().get_run_settings()['observables'],
    )
    assert fast.get_values('herfindahl')[-20:].mean() > slow.get_values('herfindahl')[-20:].mean()
    assert fast.get_values('credit.spread')[-20:].mean() < slow.get_values('credit.spread')[-20:].mean()


def test_credit_net_worth_moves_between_firms_and_banks():
    """Interest and bad debt move net worth between firms and banks, so the total changes by returns and entries."""
    trajectory = _run_observed(
        CreditNetworksSystem(40, 5).prepare().system_,
        20,
        observables=CreditNetworksSystem().get_run_settings()['observables'],
    )
    total = trajectory.get_values('bank_equity') + trajectory.get_values('firm_worth')
    changes = trajectory.get_values('returns')[1:] + trajectory.get_values('injected')[1:]
    np.testing.assert_allclose(np.diff(total), changes, rtol=1e-6, atol=1e-6)
