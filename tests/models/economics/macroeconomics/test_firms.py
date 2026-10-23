"""Test the `_firms` module of `skcomplex.models.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.models import describe
from skcomplex.models.economics.macroeconomics import toy_firm_investment

from ..conftest import _agents, _run


def test_toy_firm_investment_keeps_its_accounting_identities():
    """Test that every firm produces in proportion to its capital, stays solvent, and sums to aggregate production."""
    trajectory = _run(
        built := toy_firm_investment(50),
        100,
        observables=describe('toy_firm_investment').observables_of(built),
    )
    for frame in (1, 50, 100):
        firms = _agents(trajectory, 'firms', frame)
        assert np.allclose(firms['Y'], 0.1 * firms['K'])
        assert np.all(firms['A'] >= 0)
        assert np.all(firms['B'] >= 0)
        assert trajectory['production'][frame] == pytest.approx(firms['Y'].sum())
