"""Provide equilibrium samples of the two-dimensional Ising model shared by the field and criticality tests."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csgraph

ISING_SIZES = (16, 32, 64)
ISING_TEMPERATURES = (2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45)
BURN_IN = 200
MEASURED = 800
FLIP = 0.5


def _swendsen_wang_sweep(spins: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Return the spins after one Swendsen-Wang update of a periodic square lattice."""
    side = spins.shape[0]
    probability = 1 - np.exp(-2 * beta)
    index = np.arange(side * side).reshape(side, side)
    rows, columns = [], []
    for axis in (0, 1):
        neighbour = np.roll(spins, -1, axis=axis)
        bonded = (spins == neighbour) & (rng.random(spins.shape) < probability)
        rows.append(index[bonded])
        columns.append(np.roll(index, -1, axis=axis)[bonded])
    first, second = np.concatenate(rows), np.concatenate(columns)
    graph = sparse.csr_array((np.ones(first.size), (first, second)), shape=(side * side, side * side))
    count, labels = csgraph.connected_components(graph, directed=False)
    flips = rng.random(count) < FLIP
    return np.where(flips[labels].reshape(side, side), -spins, spins)


def _magnetizations(side: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """Return the magnetization per spin after every measured Swendsen-Wang sweep in equilibrium."""
    spins = rng.choice([-1, 1], (side, side))
    samples = []
    for sweep in range(BURN_IN + MEASURED):
        spins = _swendsen_wang_sweep(spins, 1 / temperature, rng)
        if sweep >= BURN_IN:
            samples.append(spins.mean())
    return np.asarray(samples)


@pytest.fixture(scope='session')
def ising_magnetizations() -> dict[int, list[np.ndarray]]:
    """Return equilibrium magnetizations of Ising lattices of every size at every temperature."""
    rng = np.random.default_rng(0)
    return {
        side: [_magnetizations(side, temperature, rng) for temperature in ISING_TEMPERATURES] for side in ISING_SIZES
    }
