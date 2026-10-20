"""Measure agent snapshots: order parameters, kinetic temperatures, neighbour statistics, and coarse-grained fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.spatial import cKDTree

from ..base import AnalysisError
from ._extremes import GoodnessOfFit
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


def _unit_vectors(headings: npt.ArrayLike, *, vectors: bool) -> Array:
    """Return unit vectors from angles, or normalized velocity vectors, with agents on the second to last axis."""
    array = np.asarray(headings, dtype=float)
    if not vectors:
        return np.stack([np.cos(array), np.sin(array)], axis=-1)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


@analysis(DataKind.AGENTS)
def polar_order(headings: npt.ArrayLike, *, vectors: bool = False) -> Array | float:
    """Return the polar order, the length of the mean heading, of one snapshot or of every frame.

    Args:
        headings:
            The heading angles of the agents, or one row of angles per frame, or velocity vectors with dimensions last
            when `vectors` is true.

        vectors:
            Whether the headings are velocity vectors.

    Returns:
        order:
            1 when every agent moves the same way and about `1 / sqrt(n)` for random headings, one value per frame.

    Examples:
        >>> from skcomplex.core.analysis import polar_order
        >>> polar_order([0.0, 0.0, 0.0]), round(polar_order([0.0, 3.141592653589793]), 6)
        (1.0, 0.0)
    """
    units = _unit_vectors(headings, vectors=vectors)
    order = np.linalg.norm(units.mean(axis=-2), axis=-1)
    return float(order) if np.ndim(order) == 0 else order


@analysis(DataKind.AGENTS)
def nematic_order(headings: npt.ArrayLike) -> Array | float:
    """Return the nematic order `|<exp(2 i theta)>|` of headings in a plane, which ignores the sign of a heading.

    Args:
        headings:
            The heading angles of the agents, or one row per frame.

    Returns:
        order:
            1 when every agent moves along the same axis, either way.

    Examples:
        >>> from skcomplex.core.analysis import nematic_order
        >>> round(nematic_order([0.0, 3.141592653589793]), 6)
        1.0
    """
    order = np.abs(np.mean(np.exp(2j * np.asarray(headings, dtype=float)), axis=-1))
    return float(order) if np.ndim(order) == 0 else order


@analysis(DataKind.AGENTS)
def phase_order(phases: npt.ArrayLike) -> complex | npt.NDArray[np.complex128]:
    """Return Kuramoto's complex order parameter `<exp(i theta)>` of phases or headings.

    Args:
        phases:
            The phases, or one row per frame.

    Returns:
        order:
            The complex mean, whose modulus is the synchrony and whose argument is the mean phase.

    Examples:
        >>> from skcomplex.core.analysis import phase_order
        >>> phase_order([0.0, 0.0])
        (1+0j)
    """
    order = np.asarray(np.mean(np.exp(1j * np.asarray(phases, dtype=float)), axis=-1), dtype=np.complex128)
    return complex(order) if order.ndim == 0 else order


@analysis(DataKind.AGENTS)
def milling_order(positions: npt.ArrayLike, velocities: npt.ArrayLike) -> Array | float:
    """Return the milling order, the mean normalized angular momentum of agents about their centre in a plane.

    Args:
        positions:
            The positions of the agents, one row per agent with 2 columns, or one such block per frame.

        velocities:
            The velocities, shaped like the positions.

    Returns:
        order:
            1 when every agent turns the same way about the centre, near 0 for a polarized or disordered group.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import milling_order
        >>> angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        >>> ring = np.column_stack([np.cos(angles), np.sin(angles)])
        >>> round(milling_order(ring, np.column_stack([-ring[:, 1], ring[:, 0]])), 6)
        1.0
    """
    places = np.asarray(positions, dtype=float)
    moving = _unit_vectors(velocities, vectors=True)
    relative = _unit_vectors(places - places.mean(axis=-2, keepdims=True), vectors=True)
    momentum = relative[..., 0] * moving[..., 1] - relative[..., 1] * moving[..., 0]
    order = np.abs(momentum.mean(axis=-1))
    return float(order) if np.ndim(order) == 0 else order


@analysis(DataKind.AGENTS)
def kinetic_temperature(velocities: npt.ArrayLike, mass: float = 1.0) -> Array | float:
    """Return the kinetic temperature `m <|v|²> / d` of agent velocities in `d` dimensions, with `k_B = 1`.

    Args:
        velocities:
            One row per agent with one column per dimension, or one such block per frame.

        mass:
            The mass of every agent.

    Returns:
        temperature:
            The temperature, one value per frame.

    Examples:
        >>> from skcomplex.core.analysis import kinetic_temperature
        >>> kinetic_temperature([[1.0, 0.0], [0.0, 1.0]])
        0.5
    """
    array = np.asarray(velocities, dtype=float)
    temperature = mass * np.mean(np.sum(array**2, axis=-1), axis=-1) / array.shape[-1]
    return float(temperature) if np.ndim(temperature) == 0 else temperature


@analysis(DataKind.AGENTS)
def maxwell_boltzmann_test(
    velocities: npt.ArrayLike,
    mass: float = 1.0,
    temperature: float | None = None,
) -> GoodnessOfFit:
    """Test agent speeds against the Maxwell-Boltzmann law of their dimension, a chi law of `d` degrees of freedom.

    Args:
        velocities:
            One row per agent with one column per dimension.

        mass:
            The mass of every agent.

        temperature:
            The temperature of the law, or `None` for the kinetic temperature of the velocities.

    Returns:
        fit:
            The law, the temperature, the Kolmogorov-Smirnov statistic, and its p-value.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import maxwell_boltzmann_test
        >>> velocities = np.random.default_rng(0).normal(0, 1.5, (2000, 3))
        >>> maxwell_boltzmann_test(velocities).p_value > 0.05
        True
    """
    array = np.asarray(velocities, dtype=float)
    chosen = float(kinetic_temperature(array, mass)) if temperature is None else temperature
    speeds = np.linalg.norm(array, axis=-1)
    law = stats.chi(df=array.shape[-1], scale=math.sqrt(chosen / mass))
    test = stats.kstest(speeds, law.cdf)
    return GoodnessOfFit('maxwell-boltzmann', {'temperature': chosen}, float(test.statistic), float(test.pvalue))


def _tree(positions: npt.ArrayLike, box: npt.ArrayLike | None) -> tuple[Array, cKDTree]:
    """Return the positions and their tree, periodic in a box when one is given."""
    array = np.asarray(positions, dtype=float)
    array = array.reshape(array.shape[0], -1)
    if box is None:
        return array, cKDTree(array)
    sides = np.asarray(box, dtype=float)
    return np.mod(array, sides), cKDTree(np.mod(array, sides), boxsize=sides)


@analysis(DataKind.AGENTS)
def nearest_neighbor_distances(positions: npt.ArrayLike, box: npt.ArrayLike | None = None) -> Array:
    """Return the distance from every agent to its nearest neighbour, across periodic edges in a box.

    Args:
        positions:
            One row per agent.

        box:
            The sides of a periodic box, or `None` for open space.

    Returns:
        distances:
            One distance per agent.

    Examples:
        >>> from skcomplex.core.analysis import nearest_neighbor_distances
        >>> nearest_neighbor_distances([[0.5, 0.0], [9.5, 0.0]], box=[10.0, 10.0]).tolist()
        [1.0, 1.0]
    """
    array, tree = _tree(positions, box)
    return tree.query(array, k=2)[0][:, 1]


@analysis(DataKind.AGENTS)
def clark_evans_ratio(positions: npt.ArrayLike, box: npt.ArrayLike) -> float:
    """Return the Clark-Evans ratio, the mean nearest-neighbour distance over its value for uniform random points.

    Args:
        positions:
            One row per agent in a periodic box.

        box:
            The sides of the periodic box.

    Returns:
        ratio:
            1 for random points, below 1 for clustered ones, and up to 2.15 for a triangular lattice in a plane.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import clark_evans_ratio
        >>> points = np.random.default_rng(0).random((5000, 2))
        >>> round(clark_evans_ratio(points, [1.0, 1.0]), 1)
        1.0
    """
    sides = np.asarray(box, dtype=float)
    distances = nearest_neighbor_distances(positions, sides)
    dimension = sides.size
    density = distances.size / np.prod(sides)
    ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    expected = math.gamma(1 / dimension + 1) / (density * ball) ** (1 / dimension)
    return float(distances.mean() / expected)


@analysis(DataKind.AGENTS)
def radial_distribution(
    positions: npt.ArrayLike,
    box: npt.ArrayLike,
    *,
    bins: int = 100,
    r_max: float | None = None,
) -> tuple[Array, Array]:
    """Return the radial distribution function `g(r)` of positions in a periodic box of any dimension.

    Args:
        positions:
            One row per agent, or one block of rows per frame, averaged over frames.

        box:
            The sides of the periodic box.

        bins:
            The number of distance bins.

        r_max:
            The largest distance, or `None` for half the smallest side.

    Returns:
        distribution:
            The bin centres and `g` at each, 1 for an ideal gas.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import radial_distribution
        >>> distances, g = radial_distribution(np.random.default_rng(0).random((4000, 2)) * 10, [10.0, 10.0], bins=10)
        >>> bool(np.all(np.abs(g - 1) < 0.1))
        True
    """
    sides = np.asarray(box, dtype=float)
    array = np.asarray(positions, dtype=float)
    snapshots = array if array.ndim == 3 else array[None]  # noqa: PLR2004
    reach = float(sides.min() / 2) if r_max is None else r_max
    edges = np.linspace(0, reach, bins + 1)
    counts = np.zeros(bins)
    for snapshot in snapshots:
        _, tree = _tree(snapshot, sides)
        pairs = tree.sparse_distance_matrix(tree, reach, output_type='ndarray')
        distances = pairs['v'][pairs['i'] != pairs['j']]
        counts += np.histogram(distances, bins=edges)[0]
    dimension = sides.size
    ball = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    shells = ball * (edges[1:] ** dimension - edges[:-1] ** dimension)
    size = snapshots.shape[1]
    ideal = snapshots.shape[0] * size * (size - 1) / np.prod(sides) * shells
    return (edges[1:] + edges[:-1]) / 2, counts / ideal


@analysis(DataKind.AGENTS)
def coarse_grain(
    positions: npt.ArrayLike,
    box: npt.ArrayLike,
    shape: Sequence[int],
    values: npt.ArrayLike | None = None,
) -> Array:
    """Return agents coarse-grained into a field on a grid: their number, or the sum of a value, in every cell.

    Args:
        positions:
            One row per agent in the box.

        box:
            The sides of the box.

        shape:
            The number of cells along every side.

        values:
            One value or one row of values per agent, such as unit headings for a polarization field, or `None` to
            count agents.

    Returns:
        field:
            The grid of counts, or of sums with the value components last, ready for field analyses.

    Raises:
        AnalysisError:
            The shape does not have one entry per dimension of the box.

    Examples:
        >>> from skcomplex.core.analysis import coarse_grain
        >>> coarse_grain([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [1.0, 1.0], [2, 2]).tolist()
        [[2.0, 0.0], [0.0, 1.0]]
    """
    sides = np.asarray(box, dtype=float)
    if len(shape) != sides.size:
        error_msg = f'The shape {tuple(shape)} needs one entry per dimension of the box, {sides.size}.'
        raise AnalysisError(error_msg)
    array = np.mod(np.asarray(positions, dtype=float).reshape(-1, sides.size), sides)
    cells = np.minimum((array / sides * np.asarray(shape)).astype(int), np.asarray(shape) - 1)
    flat = np.ravel_multi_index(tuple(cells.T), tuple(shape))
    total = int(np.prod(shape))
    if values is None:
        return np.bincount(flat, minlength=total).reshape(tuple(shape)).astype(float)
    weights = np.asarray(values, dtype=float).reshape(flat.size, -1)
    columns = [np.bincount(flat, weights[:, index], minlength=total) for index in range(weights.shape[1])]
    field = np.stack(columns, axis=-1).reshape(*shape, weights.shape[1])
    return field[..., 0] if weights.shape[1] == 1 else field
