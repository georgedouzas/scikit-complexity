"""Measure fluctuations: number, relative, and velocity fluctuations, Fano factors, and counts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
KINDS = ('number', 'relative', 'fano', 'kinetic-temperature', 'maxwell-boltzmann')
SETTINGS = {
    'number': {'variables', 'boxes'},
    'relative': set(),
    'fano': {'variables'},
    'kinetic-temperature': {'group', 'mass'},
    'maxwell-boltzmann': {'group', 'mass', 'temperature'},
}
"""Settings every kind reads, besides `kind`."""
DATA_KINDS = {
    'number': DataKind.FIELD,
    'relative': DataKind.ENSEMBLE,
    'fano': DataKind.SERIES,
    'kinetic-temperature': DataKind.AGENTS,
    'maxwell-boltzmann': DataKind.AGENTS,
}


def _fano_factor(values: npt.ArrayLike) -> float:
    """Return the variance over the mean of counts, raising an error when the mean is 0."""
    array = np.asarray(values, dtype=float)
    if array.mean() == 0:
        error_msg = 'The counts have mean 0, so their Fano factor is undefined. Use counts with a positive mean.'
        raise AnalysisError(error_msg)
    return float(array.var() / array.mean())


def _number_fluctuations(grids: Array, boxes: npt.ArrayLike | None) -> tuple[Array, Array, float]:
    """Return the mean and deviation of the number in square boxes of every side and the slope of their logarithms."""
    smallest = min(grids.shape[1:])
    sides = 2 ** np.arange(int(np.log2(max(smallest // 4, 1))) + 1) if boxes is None else np.asarray(boxes, dtype=int)
    means, deviations = [], []
    for side in sides:
        rows, columns = grids.shape[1] // side, grids.shape[2] // side
        trimmed = grids[:, : rows * side, : columns * side]
        counts = trimmed.reshape(grids.shape[0], rows, side, columns, side).sum(axis=(2, 4)).ravel()
        means.append(counts.mean())
        deviations.append(counts.std())
    exponent = float(np.polyfit(np.log(means), np.log(deviations), 1)[0])
    return np.asarray(means), np.asarray(deviations), exponent


def _relative_fluctuations(samples: Mapping[float, npt.ArrayLike]) -> tuple[Array, Array, float]:
    """Return the sizes, the deviation over the mean at every size, and the slope of their logarithms."""
    sizes = np.asarray(sorted(samples), dtype=np.float64)
    arrays = [np.asarray(samples[size], dtype=np.float64) for size in sorted(samples)]
    ratios = np.asarray([array.std() / array.mean() for array in arrays], dtype=np.float64)
    exponent = float(np.polyfit(np.log(sizes), np.log(ratios), 1)[0])
    return sizes, ratios, exponent


def _kinetic_temperature(velocities: npt.ArrayLike, mass: float = 1.0) -> Array:
    """Return `m <|v|²> / d` of the velocities of the agents of every frame in `d` dimensions, with `k_B = 1`."""
    array = np.asarray(velocities, dtype=float)
    return mass * np.mean(np.sum(array**2, axis=-1), axis=-1) / array.shape[-1]


def _maxwell_boltzmann(velocities: Array, mass: float, temperature: float | None) -> tuple[float, float, float]:
    """Return the temperature and the Kolmogorov-Smirnov test of the speeds of one frame against the chi law."""
    chosen = float(_kinetic_temperature(velocities, mass)) if temperature is None else temperature
    speeds = np.linalg.norm(velocities, axis=-1)
    law = stats.chi(df=velocities.shape[-1], scale=math.sqrt(chosen / mass))
    test = stats.kstest(speeds, law.cdf)
    return chosen, float(test.statistic), float(test.pvalue)


class FluctuationAnalysis(BaseAnalysis):
    """Fluctuations of counts, densities, sampled quantities, or agent velocities, of one kind.

    The kinds read fields, samples, series, or agents:

    - `'number'`: the mean and standard deviation of the number of particles of a density field in square boxes of
      growing side over every frame, and the slope of `log deviation` against `log mean`, 1/2 for independent particles
      and above for giant number fluctuations.
    - `'relative'`: the standard deviation over the mean of a quantity sampled at several system sizes, a mapping from
      size to samples, and the slope of `log ratio` against `log size`, -1/2 for independent parts.
    - `'fano'`: the variance over the mean of every series of counts, 1 for Poisson counts, above 1 for bursty and
      below 1 for regular ones.
    - `'kinetic-temperature'`: the kinetic temperature `m <|v|²> / d` of the agents of every frame, with `k_B = 1`.
    - `'maxwell-boltzmann'`: the Kolmogorov-Smirnov test of the speeds of every frame against the Maxwell-Boltzmann
      law of their dimension, a chi law of `d` degrees of freedom.

    An array of velocities holds one snapshot, one row per agent, or one snapshot per frame along its first axis. An
    array of a density field is one grid, such as `coarse_grain` returns, or a stack of grids.

    Args:
        kind:
            The fluctuation to compute, one of the kinds above.

        variables:
            The series of a trajectory of `'fano'`, or the lattice state of `'number'`, or `None` for every scalar
            series or the first lattice state.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        boxes:
            The box sides in cells, or `None` for powers of 2 up to a quarter of the smallest side.

        mass:
            The mass of every agent.

        temperature:
            The temperature of the Maxwell-Boltzmann law, or `None` for the kinetic temperature of every frame.

    Attributes:
        values_:
            The density frames, the samples by size, the series as columns, or the velocities, frames by agents by
            dimensions.

        means_:
            The mean number in boxes of every side.

        deviations_:
            The standard deviation of the number in boxes of every side.

        sizes_:
            The system sizes.

        ratios_:
            The standard deviation over the mean at every size.

        exponent_:
            The slope of the number or relative fluctuations.

        factor_:
            The Fano factor of every series.

        temperature_:
            The kinetic temperature, or the temperature of the tested law, of every frame.

        statistic_:
            The Kolmogorov-Smirnov statistic of every frame.

        p_value_:
            The p-value of every frame.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import FluctuationAnalysis
        >>> FluctuationAnalysis('fano').check().prepare([1, 3, 1, 3]).run().factor_.tolist()
        [0.5]
        >>> counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
        >>> round(FluctuationAnalysis('number').check().prepare(counts).run().exponent_, 1)
        0.5
        >>> FluctuationAnalysis('kinetic-temperature').check().prepare([[1.0, 0.0], [0.0, 1.0]]).run().temperature_
        array([0.5])
    """

    def __init__(
        self,
        kind: str = 'fano',
        variables: Sequence[str] | None = None,
        *,
        group: str | None = None,
        boxes: Sequence[int] | None = None,
        mass: float = 1.0,
        temperature: float | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.group = group
        self.boxes = boxes
        self.mass = mass
        self.temperature = temperature

    def _check_ignored(self) -> None:
        """Raise an error for a setting changed from its default that the kind does not read."""
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind'}
        ignored = [name for name, value in self.get_params().items() if name not in read and value != defaults[name]]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind])) or "no other setting"}.'
            )
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind, the mass, the temperature, and that the kind reads every changed setting."""
        if self.kind not in KINDS:
            error_msg = f'{type(self).__name__} has kind={self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        self._check_ignored()
        if self.mass <= 0 or (self.temperature is not None and self.temperature <= 0):
            error_msg = (
                f'{type(self).__name__} has mass={self.mass!r} and temperature={self.temperature!r}. Use positive '
                'values.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the chosen fluctuation reads."""
        return DATA_KINDS[self.kind]

    def _prepare_velocities(self, source: Source) -> Self:
        """Read the velocities of an agent group, or an array of one snapshot or of frames."""
        if isinstance(source, Trajectory):
            velocities = self._read_snapshots(source, self.group).velocities
            if velocities is None:
                error_msg = f'The agents of {self.group or "the first group"} have no velocities. Use moving agents.'
                raise AnalysisError(error_msg)
            self.values_ = velocities
            return self
        array = np.asarray(source, dtype=float)
        self.values_ = array[None] if array.ndim == 2 else array  # noqa: PLR2004
        return self

    def _prepare(self, source: Source | Mapping[float, npt.ArrayLike]) -> Self:
        """Read the density frames, the samples by size, the series, or the velocities of the source."""
        if self.kind == 'relative':
            if not isinstance(source, Mapping):
                error_msg = f'{type(self).__name__} of kind relative reads a mapping from size to samples.'
                raise AnalysisError(error_msg)
            self.values_ = dict(source)
            return self
        if self.kind == 'number':
            name = self.variables[0] if self.variables else None
            frames = self._read_fields(source, name)  # type: ignore[arg-type]
            self.values_ = frames if frames.ndim == 3 else frames[None]  # noqa: PLR2004
            return self
        if self.kind == 'fano':
            self.values_ = self._read_series(source, self.variables).values  # type: ignore[arg-type]
            return self
        return self._prepare_velocities(source)  # type: ignore[arg-type]

    def _run(self) -> Self:
        """Compute the fluctuations."""
        if self.kind == 'number':
            self.means_, self.deviations_, self.exponent_ = _number_fluctuations(self.values_, self.boxes)
        elif self.kind == 'relative':
            self.sizes_, self.ratios_, self.exponent_ = _relative_fluctuations(self.values_)
        elif self.kind == 'fano':
            self.factor_ = np.asarray([_fano_factor(column) for column in self.values_.T])
        elif self.kind == 'kinetic-temperature':
            self.temperature_ = _kinetic_temperature(self.values_, self.mass)
        else:
            tests = [_maxwell_boltzmann(frame, self.mass, self.temperature) for frame in self.values_]
            self.temperature_, self.statistic_, self.p_value_ = np.asarray(tests).T
        return self


@analysis(DataKind.PATH)
def event_counts(event_times: npt.ArrayLike, window: float, *, start: float = 0.0, end: float | None = None) -> Array:
    """Return the number of events in consecutive windows of equal length.

    Args:
        event_times:
            The times of the events.

        window:
            The length of every window.

        start:
            The start of the first window.

        end:
            The end of the last full window, or `None` for the last event.

    Returns:
        counts:
            The number of events in every window.

    Examples:
        >>> from skcomplex.analysis import event_counts
        >>> event_counts([0.1, 0.2, 1.5, 2.2], 1.0, end=3.0).tolist()
        [2.0, 1.0, 1.0]
    """
    times = np.asarray(event_times, dtype=float)
    last = float(times.max()) if end is None else end
    count = int((last - start) // window)
    edges = start + window * np.arange(count + 1)
    return np.histogram(times, bins=edges)[0].astype(float)
