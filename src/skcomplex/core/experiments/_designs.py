"""Give the rows of values an experiment runs, such as grids and random samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from scipy.stats import qmc

from ..errors import ValidationError
from .types import Bounds, Row

DESIGNS = ('grid', 'random', 'latin_hypercube', 'sobol')
BOUND_LENGTH = 2


def _get_rows(
    parameters: Bounds | Sequence[Mapping[str, object]],
    design: str,
    n: int,
    random_state: int | None,
    integers: Iterable[str] = (),
) -> list[Row]:
    """Return the rows of a sweep.

    They are the rows given, every combination of values, or samples within bounds.

    Args:
        parameters:
            A sequence of values per name for `'grid'`, a pair `(low, high)` or a SciPy distribution per name for the
            sampled designs, or a sequence of rows, each a mapping from name to value, run as they are.

        design:
            `'grid'`, `'random'`, `'latin_hypercube'`, or `'sobol'`.

        n:
            The number of rows of a sampled design.

        random_state:
            The seed of the samples.

        integers:
            The names whose samples are integers, drawn from the integers within their pair of bounds.

    Returns:
        rows:
            One mapping from name to value per row.

    Raises:
        ValidationError:
            The design is unknown, rows are given with a sampled design, or the values or bounds do not fit the
            design.
    """
    if design not in DESIGNS:
        error_msg = f'Unknown design {design!r}. Use one of {", ".join(DESIGNS)}.'
        raise ValidationError(error_msg)
    if not isinstance(parameters, Mapping):
        if design != 'grid':
            error_msg = f'A sweep runs given rows as they are, so it takes no {design} design.'
            raise ValidationError(error_msg)
        for row in parameters:
            if not isinstance(row, Mapping):
                error_msg = f'Every row of a sweep must be a mapping from name to value, not {row!r}.'
                raise ValidationError(error_msg)
        return [dict(row) for row in parameters]
    if design == 'grid':
        return _get_grid_rows(parameters)
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
        error_msg = f'n = {n!r} must be a nonnegative integer.'
        raise ValidationError(error_msg)
    bounds = _check_bounds(parameters)
    if design == 'random':
        uniform = np.random.default_rng(random_state).random((n, len(bounds)))
    elif design == 'latin_hypercube':
        uniform = qmc.LatinHypercube(d=len(bounds), seed=random_state).random(n)
    else:
        uniform = qmc.Sobol(d=len(bounds), seed=random_state).random(n)
    return _scale(bounds, uniform, integers)


def _get_grid_rows(values: Bounds) -> list[Row]:
    """Return one row per combination of the values of every name.

    The last name varies fastest.

    Args:
        values:
            A sequence of values per name.

    Returns:
        rows:
            One row per combination.

    Raises:
        ValidationError:
            The values of a name are not a sequence.
    """
    grid: dict[str, list[object]] = {}
    for name, items in values.items():
        if isinstance(items, str) or not isinstance(items, Iterable):
            error_msg = f'A grid takes a sequence of values per name, but {name} has {items!r}.'
            raise ValidationError(error_msg)
        grid[name] = list(items)
    names = list(grid)
    return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*grid.values())]


def _check_bounds(bounds: Bounds, *, distributions: bool = True) -> dict[str, object]:
    """Return the bounds by name.

    It raises an error for one that is neither a pair nor an allowed distribution.

    Args:
        bounds:
            A pair `(low, high)`, or a SciPy distribution when allowed, per name.

        distributions:
            Whether SciPy distributions are allowed.

    Returns:
        bounds:
            The bounds by name.

    Raises:
        ValidationError:
            A bound is neither a pair nor an allowed distribution.
    """
    checked = dict(bounds)
    for name, bound in checked.items():
        pair = isinstance(bound, Sequence) and not isinstance(bound, str) and len(bound) == BOUND_LENGTH
        if pair or (distributions and hasattr(bound, 'ppf')):
            continue
        kinds = 'a pair (low, high) or a SciPy distribution' if distributions else 'a pair (low, high)'
        error_msg = f'The bounds of {name} must be {kinds}, not {bound!r}.'
        raise ValidationError(error_msg)
    return checked


def _get_pairs(bounds: Bounds) -> dict[str, tuple[float, float]]:
    """Return the pairs `(low, high)` of bounds by name.

    Args:
        bounds:
            A pair `(low, high)` per name.

    Returns:
        pairs:
            The pairs of floats by name.
    """
    pairs = {}
    for name, bound in _check_bounds(bounds, distributions=False).items():
        if isinstance(bound, Sequence):
            low, high = (float(item) for item in bound)
            pairs[name] = (low, high)
    return pairs


def _scale(
    bounds: Mapping[str, object],
    uniform: npt.NDArray[np.float64],
    integers: Iterable[str] = (),
) -> list[Row]:
    """Return rows of uniform samples in the unit cube mapped to bounds.

    They are mapped to pairs of bounds or through quantile functions. A name among the integers takes each integer
    within its pair of bounds with equal chance, both bounds included.

    Args:
        bounds:
            A pair `(low, high)` or a SciPy distribution per name.

        uniform:
            The samples in the unit cube, one row per sample and one column per name.

        integers:
            The names whose samples are integers.

    Returns:
        rows:
            One row per sample.
    """
    columns = {}
    integral = set(integers)
    for index, (name, bound) in enumerate(bounds.items()):
        if hasattr(bound, 'ppf'):
            columns[name] = np.asarray(bound.ppf(uniform[:, index]), dtype=float)
        elif isinstance(bound, Sequence) and name in integral:
            low, high = np.ceil(float(bound[0])), np.floor(float(bound[1]))
            columns[name] = np.minimum(np.floor(low + uniform[:, index] * (high - low + 1)), high)
        elif isinstance(bound, Sequence):
            low, high = (float(item) for item in bound)
            columns[name] = low + uniform[:, index] * (high - low)
    return [{name: float(column[row]) for name, column in columns.items()} for row in range(uniform.shape[0])]


def _get_sensitivity_rows(bounds: Bounds, method: str, n: int, random_state: int | None) -> list[Row]:
    """Return the rows a sensitivity method reads.

    SALib gives Morris trajectories and Saltelli's sample. - `'oat'`: two rows per name, at its lower and then at its
    upper bound, with the others at their midpoints. - `'morris'`: `n` Morris trajectories on four levels, `n * (names +
    1)` rows. - `'sobol'`: Saltelli's sample of `n` base points, best a power of 2, `n * (names + 2)` rows.

    Args:
        bounds:
            A pair `(low, high)` per name.

        method:
            `'oat'`, `'morris'`, or `'sobol'`.

        n:
            The number of Morris trajectories or of base points.

        random_state:
            The seed of the samples.

    Returns:
        rows:
            One mapping from name to value per row.
    """
    pairs = _get_pairs(bounds)
    if method == 'oat':
        middle = {name: (low + high) / 2 for name, (low, high) in pairs.items()}
        return [{**middle, name: bound} for name, pair in pairs.items() for bound in pair]
    names = list(pairs)
    problem = {'num_vars': len(names), 'names': names, 'bounds': [list(pairs[name]) for name in names]}
    if method == 'morris':
        samples = morris_sample.sample(problem, n, num_levels=4, seed=random_state)
    else:
        samples = sobol_sample.sample(problem, n, calc_second_order=False, seed=random_state)
    return [dict(zip(names, map(float, sample), strict=True)) for sample in samples]
