"""Test the `_min_proteins` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np

from skcomplex.core.analysis import stability
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.systems_biology import MinProteinSkeletonSystem

from ..conftest import _run_with_solver


def test_min_protein_skeleton_conserves_both_proteins_and_forms_oscillating_bands():
    """Test that the Min skeleton system conserves MinD and MinE while an oscillatory instability forms bands."""
    system = MinProteinSkeletonSystem(64).prepare().system_
    relation = stability(system, kind='dispersion')
    unstable = relation.growth_rates > 0
    assert unstable.any()
    assert np.abs(relation.frequencies[unstable]).max() > 0.01
    trajectory = _run_with_solver(system, 200, 10.0, ScipyODESolver(method='BDF', jacobian='exact'))
    states = {name: trajectory.get_values(f'cell.{name}') for name in ('md', 'mde', 'cdt', 'cdd', 'ce')}
    total_D = (states['md'] + states['mde'] + states['cdt'] + states['cdd']).sum(axis=1)
    total_E = (states['mde'] + states['ce']).sum(axis=1)
    np.testing.assert_allclose(total_D, total_D[0], rtol=1e-6)
    np.testing.assert_allclose(total_E, total_E[0], rtol=1e-6)
    assert np.ptp(states['md'][-1]) > 2 * states['md'][0].mean()
