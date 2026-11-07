"""Test the `_catalysis` module of `skcomplex.systems.economics.innovation`."""

import numpy as np

from skcomplex.core.analysis import autocatalytic_sets
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.economics.innovation import TapCatalysisSystem


def _final(seed, **arguments):
    """Return the network of the last frame of a TAP run."""
    trajectory = (
        Simulation(TapCatalysisSystem(**arguments).prepare().system_, random_state=seed)
        .prepare()
        .run(until=400)
        .trajectory_
    )
    return trajectory.network(-1)


def test_tap_grows_one_production_function_per_new_good_from_older_parents():
    """Test that the growth stops at the number of goods, and every new good is produced from up to K older goods."""
    network = _final(0, goods=200, parents=3)
    functions = network.hyperedges
    sizes = (functions >= 0).sum(axis=1)
    products = functions[np.arange(len(functions)), sizes - 1]
    assert len(network.nodes) == 200
    assert functions.shape == (190, 4)
    assert set(sizes.tolist()) <= {2, 3, 4}
    assert sorted(products.tolist()) == list(range(10, 200))
    assert all((row[: size - 1] < row[size - 1]).all() for row, size in zip(functions, sizes, strict=True))


def test_catalysis_joins_a_share_pc_of_the_ordered_pairs():
    """Test that the catalysis edges join about a share `pc` of the ordered pairs of goods."""
    network = _final(1, goods=300, pc=0.02)
    share = len(network.sources) / (300 * 299 - 10 * 9)
    assert 0.017 < share < 0.023


def test_autocatalytic_sets_appear_as_the_catalysis_probability_rises():
    """Test that large RAFs never appear at a low catalysis probability and always at a high one."""
    low = [autocatalytic_sets(_final(seed, goods=300, pc=0.004)) for seed in range(4)]
    high = [autocatalytic_sets(_final(seed, goods=300, pc=0.04)) for seed in range(4)]
    assert max(analysis.fractions[0] for analysis in low) < 0.5
    assert min(analysis.fractions[0] for analysis in high) > 0.9
