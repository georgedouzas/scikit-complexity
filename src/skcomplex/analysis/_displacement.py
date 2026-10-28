"""Measure the mean squared displacement of particles, its scaling exponent, and their diffusion coefficient."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import tidynamics

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis, Source

FIT_POINTS = 20


class DisplacementAnalysis(BaseAnalysis):
    """Mean squared displacement of particles over all lags, its scaling exponent, and the diffusion coefficient.

    The displacement of every particle is computed for all lags at once with the FFT of `tidynamics`, so positions
    must be unwrapped across periodic boundaries, as a trajectory gives them.

    Args:
        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        dt:
            The time between frames.

        lags:
            The lags in frames to keep, or `None` for every lag from 1.

        fit_range:
            The smallest and largest lags in time of the fits, or `None` for one frame to a quarter of the run.

    Attributes:
        positions_:
            The unwrapped positions, frames by particles by dimensions.

        lags_:
            The lags in time.

        msd_:
            The mean squared displacement at every lag.

        exponent_:
            The slope of `log msd` against `log lag` over the fit range, 1 for normal diffusion, below 1 for
            subdiffusion, and above 1 for superdiffusion.

        diffusion_:
            The coefficient `D` of `msd = 2 d D lag` fitted over the fit range in `d` dimensions.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import DisplacementAnalysis
        >>> walks = np.cumsum(np.random.default_rng(0).standard_normal((2000, 500)), axis=0)
        >>> displacement = DisplacementAnalysis().check().prepare(walks).run()
        >>> round(displacement.exponent_, 1), round(displacement.diffusion_, 1)
        (1.0, 0.5)
    """

    def __init__(
        self,
        group: str | None = None,
        *,
        dt: float = 1.0,
        lags: Sequence[int] | None = None,
        fit_range: tuple[float, float] | None = None,
    ) -> None:
        self.group = group
        self.dt = dt
        self.lags = lags
        self.fit_range = fit_range

    def _check(self) -> Self:
        """Validate the time step and the lags."""
        if self.dt <= 0:
            error_msg = f'{type(self).__name__} has dt={self.dt!r}. Use a positive time step.'
            raise AnalysisError(error_msg)
        if self.lags is not None and min(self.lags, default=1) < 1:
            error_msg = f'{type(self).__name__} has lags={self.lags!r}. Use lags of 1 frame or more.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the agents kind."""
        return DataKind.AGENTS

    def _prepare(self, source: Source) -> Self:
        """Read the unwrapped positions of an agent group, or an array with time as the first axis."""
        if isinstance(source, Trajectory):
            positions = self._read_snapshots(source, self.group).positions
            if positions is None:
                error_msg = f'The agents of {self.group or "the first group"} have no positions. Use agents in a space.'
                raise AnalysisError(error_msg)
            self.positions_ = self._particle_axes(positions)
            return self
        self.positions_ = self._particle_axes(source)
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the mean squared displacement and fit its exponent and the diffusion coefficient."""
        array = self.positions_
        msd = np.mean([tidynamics.msd(array[:, particle]) for particle in range(array.shape[1])], axis=0)[1:]
        all_lags = np.arange(1, array.shape[0], dtype=np.float64) * self.dt
        chosen = self._fit_lags(all_lags, self.fit_range)
        self.exponent_ = float(np.polyfit(np.log(all_lags[chosen]), np.log(msd[chosen]), 1)[0])
        slope = float(np.polyfit(all_lags[chosen], msd[chosen], 1)[0])
        self.diffusion_ = slope / (2 * array.shape[2])
        if self.lags is None:
            self.lags_, self.msd_ = all_lags, msd
            return self
        steps = np.asarray(self.lags, dtype=int)
        self.lags_, self.msd_ = steps * self.dt, msd[steps - 1]
        return self

    @staticmethod
    def _particle_axes(positions: npt.ArrayLike) -> Array:
        """Return positions as frames by particles by dimensions."""
        array = BaseAnalysis._read_real(positions, 'positions')
        return array.reshape(array.shape[0], array.shape[1] if array.ndim > 1 else 1, -1)

    @staticmethod
    def _fit_lags(lags: Array, fit_range: tuple[float, float] | None) -> npt.NDArray[np.intp]:
        """Return the indices of about 20 lags spaced logarithmically within the fit range."""
        low, high = (lags[0], lags[-1] / 4) if fit_range is None else fit_range
        inside = np.flatnonzero((lags >= low) & (lags <= high))
        if inside.size < 2:  # noqa: PLR2004
            error_msg = f'The fit range {fit_range} holds fewer than 2 lags. Widen it within {lags[0]} to {lags[-1]}.'
            raise AnalysisError(error_msg)
        chosen = np.unique(np.geomspace(1, inside.size, FIT_POINTS).astype(int) - 1)
        return inside[chosen]
