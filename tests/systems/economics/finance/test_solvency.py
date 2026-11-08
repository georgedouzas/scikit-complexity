"""Test the `_solvency` module of `skcomplex.systems.economics.finance`."""

import numpy as np
import pytest
from scipy.optimize import linprog

from skcomplex.systems.economics.finance import DefaultCascadeSystem, EisenbergNoeClearingSystem

from ..conftest import _agents, _run


def test_default_cascade_spreads_along_a_chain_of_exposures():
    """Test a chain by hand: each default costs the next creditor its claim, and a recovery rate stops the cascade."""
    chain = [[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]
    full = _run(DefaultCascadeSystem(chain, [1.0, 4.0, 4.0], threshold=0.5), 5)
    assert full.values('cascade.defaults')[-1] == 3
    assert _agents(full, 'banks', 5)['equity'][1:] == pytest.approx([-1.0, -1.0])
    partial = _run(DefaultCascadeSystem(chain, [1.0, 4.0, 4.0], threshold=0.5, recovery=0.5), 5)
    assert partial.values('cascade.defaults')[-1] == 1
    assert _agents(partial, 'banks', 5)['equity'][1] == pytest.approx(1.5)


def test_default_cascade_has_the_contagion_window_of_gai_and_kapadia():
    """Test that cascades from one bank are large at intermediate degrees and small for sparse or dense networks."""

    def extent(degree):
        shares = [
            _run(DefaultCascadeSystem(degree=degree, random_state=seed, shocked=(seed,)), 30).values(
                'cascade.defaults',
            )[-1]
            for seed in range(6)
        ]
        return np.mean(shares) / 250

    assert extent(4.0) > 0.4
    assert extent(0.5) < 0.05
    assert extent(12.0) < 0.05


def test_eisenberg_noe_clearing_reaches_the_greatest_clearing_vector():
    """Test that the payments solve the linear programme of the clearing vector and default only short banks."""
    system = EisenbergNoeClearingSystem().prepare().system_
    trajectory = _run(system, 60)
    matrix = np.asarray(system.interactions[0].liabilities)
    external = system.elements[0].values['external']
    owed = matrix.sum(axis=1)
    shares = np.divide(matrix, owed[:, None], out=np.zeros_like(matrix), where=owed[:, None] > 0)
    size = len(owed)
    solution = linprog(
        -np.ones(size),
        A_ub=np.eye(size) - shares.T,
        b_ub=external,
        bounds=list(zip(np.zeros(size), owed, strict=True)),
    )
    payments = _agents(trajectory, 'banks', 60)['payment']
    assert payments == pytest.approx(solution.x, abs=1e-6)
    assert np.all(np.diff(trajectory.values('clearing.shortfall')[1:]) >= -1e-9)
    assert trajectory.values('clearing.defaults')[-1] > 0
