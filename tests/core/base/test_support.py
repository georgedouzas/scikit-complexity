"""Test the `_support` module of `skcomplex.core.base`."""

from skcomplex.core.base import NoiseKind
from skcomplex.engine.solvers import EulerMaruyama, Heun, Milstein


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyama.supports.noise) == set(NoiseKind)
    assert set(Heun.supports.noise) == set(NoiseKind)
    assert set(Milstein.supports.noise) == {NoiseKind.DIAGONAL}
