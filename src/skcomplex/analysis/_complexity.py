"""Place series and fields on the complexity-entropy plane of their ordinal patterns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import ordpy

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._entropy import TIES, EntropyAnalysis
from ._kinds import DataKind


class ComplexityAnalysis(BaseAnalysis):
    """Place of every series of a source, or of every frame of a field, on Rosso's complexity-entropy plane.

    The entropy is the normalized permutation entropy of the ordinal patterns, and the complexity is that entropy times
    the Jensen-Shannon divergence of the patterns from uniform ones, computed with `ordpy`. Noise has entropy near 1 and
    complexity near 0, regular series both near 0, and chaos lies in between with high complexity.

    Args:
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

    Attributes:
        values_:
            The series, one per column, or the frames of the field.

        labels_:
            The label of every series, or of the field.

        entropy_:
            The normalized permutation entropy of every series or frame.

        complexity_:
            The statistical complexity of every series or frame.

        minimum_:
            The curve of least complexity, one row of entropy and complexity per point.

        maximum_:
            The curve of largest complexity, one row of entropy and complexity per point.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ComplexityAnalysis
        >>> noise = np.random.default_rng(0).standard_normal(50_000)
        >>> plane = ComplexityAnalysis(dimension=4).check().prepare(noise).run()
        >>> plane.entropy_.round(2).tolist(), plane.complexity_.round(2).tolist()
        ([1.0], [0.0])
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        dimension: int | tuple[int, int] = 6,
        delay: int = 1,
        ties: str = 'order',
        random_state: int | None = None,
    ) -> None:
        self.variables = variables
        self.dimension = dimension
        self.delay = delay
        self.ties = ties
        self.random_state = random_state

    def _check(self) -> Self:
        """Validate the ties, the pattern length or window, and the delay."""
        if self.ties not in TIES:
            error_msg = f'{type(self).__name__} has ties={self.ties!r}. Use one of {", ".join(TIES)}.'
            raise AnalysisError(error_msg)
        sizes = self.dimension if isinstance(self.dimension, tuple) else (self.dimension,)
        if any(int(size) < 1 for size in sizes) or self.delay < 1:
            error_msg = (
                f'{type(self).__name__} has dimension={self.dimension!r} and delay={self.delay!r}. Use 1 or more.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind for windows of a field, and the series kind otherwise."""
        return DataKind.FIELD if isinstance(self.dimension, tuple) else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series of the source as columns, or the frames of a field for windows."""
        if self._get_data_kind() == DataKind.FIELD:
            name = self.variables[0] if self.variables else None
            frames = self._read_fields(source, name)
            self.values_ = frames[None] if frames.ndim == 2 else frames  # noqa: PLR2004
            self.labels_ = [name or 'field']
            return self
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the place of every series or frame on the plane, and the bounds of the plane."""
        dx, dy = (self.dimension, 1) if isinstance(self.dimension, int) else self.dimension
        items = self.values_ if self._get_data_kind() == DataKind.FIELD else self.values_.T
        places = [
            ordpy.complexity_entropy(
                EntropyAnalysis._break_ties(item, self.ties, self.random_state),
                dx=dx,
                dy=dy,
                taux=self.delay,
                tauy=self.delay,
            )
            for item in items
        ]
        self.entropy_ = np.asarray([entropy for entropy, _ in places], dtype=float)
        self.complexity_ = np.asarray([complexity for _, complexity in places], dtype=float)
        self.minimum_ = np.asarray(ordpy.minimum_complexity_entropy(dx=dx, dy=dy), dtype=float)
        self.maximum_ = np.asarray(ordpy.maximum_complexity_entropy(dx=dx, dy=dy), dtype=float)
        return self
