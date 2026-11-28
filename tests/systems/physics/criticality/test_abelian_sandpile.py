"""Test the `_abelian_sandpile` module of `skcomplex.systems.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.core.errors import AnalysisError
from skcomplex.systems.physics.criticality import ForestFireSystem, SandpileSystem, abelian_sandpile

GRAINS = 100_000
TRANSIENT = 10_000


def test_abelian_sandpile_toppling_matrix_of_three_by_three():
    """Test that the toppling matrix of a 3 x 3 lattice holds 4 on its diagonal and minus one for every neighbour."""
    toppling = abelian_sandpile(SandpileSystem((3, 3))).toppling.toarray()
    assert toppling.shape == (9, 9)
    assert np.diag(toppling).tolist() == [4.0] * 9
    assert int((toppling == -1).sum()) == 24
    assert toppling[4, [1, 3, 5, 7]].tolist() == [-1.0] * 4


def test_abelian_sandpile_green_function_inverts_the_toppling_matrix():
    """Test that the Green function of an 8 x 8 lattice is symmetric, positive, and the inverse of its matrix."""
    system = SandpileSystem((8, 8))
    green = abelian_sandpile(system, kind='green').green
    toppling = abelian_sandpile(system).toppling
    np.testing.assert_allclose(green, green.T)
    assert (green > 0).all()
    np.testing.assert_allclose(toppling @ green, np.eye(64), atol=1e-10)


def test_abelian_sandpile_mean_size_grows_as_the_squared_side():
    """Test that the mean avalanche sizes of the lattices of side 32 and 64 grow by a factor near 4."""
    small = abelian_sandpile(SandpileSystem((32, 32)), kind='mean-size').mean_size
    large = abelian_sandpile(SandpileSystem((64, 64)), kind='mean-size').mean_size
    assert large / small == pytest.approx(4, rel=0.1)


@pytest.mark.slow
def test_abelian_sandpile_mean_size_agrees_with_a_simulated_pile():
    """Test that the topplings per grain of a 16 x 16 pile after its transient lie within 5 percent of the mean size."""
    system = SandpileSystem((16, 16)).prepare()
    events = system.simulate(until=TRANSIENT + GRAINS, dt=TRANSIENT + GRAINS, random_state=0).events
    topplings = sum(event.size for event in events if event.time >= TRANSIENT) / GRAINS
    assert topplings == pytest.approx(abelian_sandpile(system, kind='mean-size').mean_size, rel=0.05)


def test_abelian_sandpile_refuses_other_systems_kinds_and_large_inverses():
    """Test that random neighbours, other systems, unknown kinds, and large dense inverses are refused."""
    with pytest.raises(AnalysisError, match='change at every toppling'):
        abelian_sandpile(SandpileSystem(neighbors='random'))
    with pytest.raises(AnalysisError, match='SandpileSystem'):
        abelian_sandpile(ForestFireSystem())
    with pytest.raises(AnalysisError, match='toppling, green, mean-size'):
        abelian_sandpile(SandpileSystem(), kind='inverse')
    with pytest.raises(AnalysisError, match="kind='mean-size'"):
        abelian_sandpile(SandpileSystem((128, 128)), kind='green')
