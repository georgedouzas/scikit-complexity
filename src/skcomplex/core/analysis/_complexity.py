"""Place series and fields on the complexity-entropy plane of their ordinal patterns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import ordpy

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import SNAPSHOT_DIMENSIONS, _read_fields, _read_series
from ._entropy import TIES, _break_ties
from .types import Source

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class ComplexityResult:
    """Place of every series of a source, or of every frame of a field, on Rosso's complexity-entropy plane.

    Args:
        values:
            The series, one per column, or the frames of the field.

        labels:
            The label of every series, or of the field.

        entropy:
            The normalized permutation entropy of every series or frame.

        complexity:
            The statistical complexity of every series or frame.

        minimum:
            The curve of least complexity, one row of entropy and complexity per point.

        maximum:
            The curve of largest complexity, one row of entropy and complexity per point.
    """

    values: Array
    labels: list[str]
    entropy: Array
    complexity: Array
    minimum: Array
    maximum: Array


def complexity(
    source: Source,
    *,
    variables: Sequence[str] | None = None,
    dimension: int | tuple[int, int] = 6,
    delay: int = 1,
    ties: str = 'order',
    random_state: int | None = None,
) -> ComplexityResult:
    """Return the place of every series of a source, or of every frame of a field, on Rosso's complexity-entropy plane.

    The entropy is the normalized permutation entropy of the ordinal patterns, and the complexity is that entropy times
    the Jensen-Shannon divergence of the patterns from uniform ones, computed with `ordpy`. Noise has entropy near 1 and
    complexity near 0, regular series both near 0, and chaos lies in between with high complexity.

    Args:
        source:
            The trajectory, an array with one row per time and one column per series, or the frames of a field.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        dimension:
            The length of the patterns of a series, or the window `(columns, rows)` of the patterns of a field.

        delay:
            The number of steps between the values of a pattern.

        ties:
            `'order'` to rank equal values by position, or `'random'` to rank them at random.

        random_state:
            The seed of the random ranks of equal values.

    Returns:
        result:
            The values read, the place of every series or frame on the plane, and the bounds of the plane.

    Raises:
        AnalysisError:
            The ties are unknown, or the pattern length, the window, or the delay is below 1.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import complexity
        >>> noise = np.random.default_rng(0).standard_normal(50_000)
        >>> plane = complexity(noise, dimension=4)
        >>> plane.entropy.round(2).tolist(), plane.complexity.round(2).tolist()
        ([1.0], [0.0])
    """
    if ties not in TIES:
        error_msg = f'complexity has ties={ties!r}. Use one of {", ".join(TIES)}.'
        raise AnalysisError(error_msg)
    sizes = dimension if isinstance(dimension, tuple) else (dimension,)
    if any(int(size) < 1 for size in sizes) or delay < 1:
        error_msg = f'complexity has dimension={dimension!r} and delay={delay!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if isinstance(dimension, tuple):
        name = variables[0] if variables else None
        frames = _read_fields(source, name)
        values = frames[None] if frames.ndim == SNAPSHOT_DIMENSIONS else frames
        labels = [name or 'field']
        items = list(values)
    else:
        series = _read_series(source, variables)
        values, labels = series.values, series.labels
        items = list(values.T)
    dx, dy = (dimension, 1) if isinstance(dimension, int) else dimension
    places = [
        ordpy.complexity_entropy(_break_ties(item, ties, random_state), dx=dx, dy=dy, taux=delay, tauy=delay)
        for item in items
    ]
    return ComplexityResult(
        values,
        labels,
        np.asarray([entropy for entropy, _ in places], dtype=float),
        np.asarray([measure for _, measure in places], dtype=float),
        np.asarray(ordpy.minimum_complexity_entropy(dx=dx, dy=dy), dtype=float),
        np.asarray(ordpy.maximum_complexity_entropy(dx=dx, dy=dy), dtype=float),
    )
