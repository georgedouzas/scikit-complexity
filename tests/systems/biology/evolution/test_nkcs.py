"""Test the `_nkcs` module of `skcomplex.systems.biology.evolution`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.evolution import NKCSSystem


def _late_moves(k: int, seed: int) -> float:
    """Return the accepted mutations of the last 100 of 300 steps of an NKCS ecosystem."""
    trajectory = simulate(NKCSSystem(k=k, seed=seed, random_state=seed).prepare().system_, until=300, random_state=seed)
    agents = trajectory
    return float(sum(agents.get_agents('species', frame).values['moves'].sum() for frame in range(200, 301)))


def test_rugged_landscapes_freeze_and_smooth_ones_stay_chaotic():
    """Test Kauffman and Johnsen: for `k` far above `c s` the species freeze, for `k = 0` they keep adapting."""
    frozen = [_late_moves(8, seed) for seed in range(4)]
    chaotic = [_late_moves(0, seed) for seed in range(4)]
    assert sum(frozen) == 0
    assert sum(chaotic) > 0
