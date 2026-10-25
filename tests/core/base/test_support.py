"""Test the `_support` module of `skcomplex.core.base`."""

from skcomplex.core.base import NoiseKind
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver, MilsteinSolver


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyamaSolver.supports.noise) == set(NoiseKind)
    assert set(HeunSolver.supports.noise) == set(NoiseKind)
    assert set(MilsteinSolver.supports.noise) == {NoiseKind.DIAGONAL}
