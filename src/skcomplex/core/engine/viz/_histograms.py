"""Draw the variables of agents and the states of groups without a space.

They are drawn as histograms and statistics.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence, Sized

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.spaces import BaseSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...specs import AgentSnapshotSpec
from ..problems import BaseProblemSpec, FrameSpec
from ._agents import AgentsVisual
from ._base import SERIES_COLORS, BaseVisual

BINS = 20
MAX_LEVELS = 50
MAX_AGENT_SERIES = 10
STATISTICS = ('mean', 'std', 'distinct')
NUMERIC_KINDS = frozenset('biuf')


class HistogramVisual(AgentsVisual):
    """Visual that logs histograms and statistics of agents or groups without a space.

    It logs the traits of the agents or the states of the group. For every trait of the agents, a number per agent is
    taken: the value of a numeric trait, the total of a row of numbers such as the ones of a genome, or the size of a
    collection such as a vocabulary. The visual logs the histogram of these numbers as a `BarChart` under
    `<path>/<trait>/histogram`, over a range that widens to every value logged so far, with one bar per value when the
    values are few integers, and their mean and standard deviation as series under `<path>/<trait>/mean` and
    `<path>/<trait>/std`. Integer numbers, such as type labels, also log their number of distinct values under
    `<path>/<trait>/distinct`, and a group of at most ten agents logs the number of every agent as its own series under
    `<path>/<trait>/agents`. Traits of other kinds are left out. The states of the members of a group are logged the
    same way, one histogram per state.

    Args:
        color_by:
            The trait whose histogram the first view shows, or `None` for the first trait of the agents.

        color_range:
            The starting range of the histogram of the `color_by` trait, which still widens to every logged value, or
            `None` for a range widened to every logged value. The other traits always bin by their own range.

        radius:
            Not used, kept for the options of agents visuals.

        bins:
            The number of bins of the histograms.

        shown:
            The series of the colored trait that the first view shows, among `'agents'`, `'mean'`, `'std'`, and
            `'distinct'`.

        histogram_first:
            Whether the histogram view comes before the time series view of the colored trait, for traits whose
            distribution matters more than its statistics, such as clustered opinions.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        spans_:
            The range of the histogram of every trait logged so far, by entity path.

    Examples:
        >>> from skcomplex.core.engine.viz import HistogramVisual
        >>> HistogramVisual(color_by='fitness').bins
        20
    """

    def __init__(
        self,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        radius: float | None = None,
        *,
        bins: int = BINS,
        shown: Sequence[str] = ('agents', 'mean', 'std'),
        histogram_first: bool = False,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        super().__init__(color_by=color_by, color_range=color_range, radius=radius, path=path, recording=recording)
        self.bins = bins
        self.shown = shown
        self.histogram_first = histogram_first

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the agents or group, forget the old histogram ranges, and set the views.

        The old ranges are those of earlier runs.
        """
        self.spans_: dict[str, tuple[float, float]] = {}
        super()._prepare(context)

    def _log_trait(
        self,
        recording: rr.RecordingStream,
        path: str,
        trait: str,
        numbers: npt.NDArray[np.float64],
    ) -> None:
        """Log the histogram, the mean, and the standard deviation of one trait.

        They are those of the numbers of the trait. The histogram of a trait bins by the range of its own values,
        widened to every value logged so far. Only the `color_by` trait starts from `color_range`, so a trait whose
        values fall outside it, such as a negative integer trait, keeps every value in its bins.

        Args:
            recording:
                The Rerun recording.

            path:
                The entity path of the trait.

            trait:
                The name of the trait.

            numbers:
                One number per agent or member.
        """
        if path not in self.spans_:
            for index, statistic in enumerate(STATISTICS):
                style = rr.SeriesLines(colors=SERIES_COLORS[index], names=statistic)
                recording.log(f'{path}/{statistic}', style, static=True)
        start = self.spans_.get(path)
        if start is None and self.color_range is not None and trait == self.color_by:
            start = self.color_range
        span = BaseVisual._get_widened(start, numbers)
        self.spans_[path] = span
        counts, centres, width = self._get_histogram(numbers, span, self.bins)
        recording.log(f'{path}/histogram', rr.BarChart(counts, abscissa=centres, widths=np.full(len(counts), width)))
        recording.log(f'{path}/mean', rr.Scalars(float(numbers.mean())))
        recording.log(f'{path}/std', rr.Scalars(float(numbers.std())))
        if np.all(numbers == np.round(numbers)):
            recording.log(f'{path}/distinct', rr.Scalars(float(np.unique(numbers).size)))
        if numbers.size <= MAX_AGENT_SERIES:
            recording.log(f'{path}/agents', rr.Scalars(numbers))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the histogram and the statistics of every state of a group at one frame.

        The states are those of the members of the group.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled group, with its path and the dotted names of its states.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            numbers = self._get_numbers_of(frame.get_value(name))
            if numbers is not None and numbers.size:
                self._log_trait(recording, f'{element.path}/{state}', state, numbers)

    def _log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshotSpec,
        space: BaseSpace | None,
    ) -> None:
        """Log the histogram and the statistics of every trait of agents at one frame.

        The agents are those of a group.

        Args:
            recording:
                The Rerun recording.

            path:
                The path of the agent group.

            snapshot:
                The agents alive at the frame.

            space:
                The space the agents live in, `None` for this visual.
        """
        if not len(snapshot.ids):
            return
        for trait, values in snapshot.values.items():
            numbers = self._get_numbers_of(values)
            if numbers is not None and numbers.size:
                self._log_trait(recording, f'{path}/{trait}', trait, numbers)

    def _get_views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the views of the traits.

        They show the statistics of one trait over time, its histogram, and the statistics of every trait.

        Args:
            element:
                The compiled agents or group, with their path and the names of their states.

            problem:
                The compiled problem.

        Returns:
            views:
                A time series view of the shown series of `color_by`, or of the first state, and a bar chart view of
                its histogram, in the chosen order, and a time series view of the statistics of every trait.
        """
        states = list(element.states)
        if not states:
            return []
        path = element.path
        trait = self.color_by if self.color_by is not None else states[0]
        contents = [f'+ /{path}/*/mean', f'+ /{path}/*/std']
        series = rrb.TimeSeriesView(
            origin='/',
            contents=[f'+ /{path}/{trait}/{name}' for name in self.shown],
            name=f'{path} {trait}',
        )
        histogram = rrb.BarChartView(origin=f'{path}/{trait}/histogram', name=f'{path} {trait} histogram')
        first = [histogram, series] if self.histogram_first else [series, histogram]
        return [*first, rrb.TimeSeriesView(origin='/', contents=contents, name=f'{path} traits')]

    @staticmethod
    def _get_numbers_of(values: object) -> npt.NDArray[np.float64] | None:
        """Return one number per agent.

        It is the value, the total of a row of numbers, or the size of a collection.
        """
        array = np.asarray(values)
        if array.dtype.kind in NUMERIC_KINDS:
            numbers = array.astype(float)
            return numbers if numbers.ndim == 1 else numbers.reshape(len(numbers), -1).sum(axis=1)
        items = list(np.ravel(array)) if array.ndim else [array.item()]
        if items and all(isinstance(item, Sized) and not isinstance(item, str) for item in items):
            return np.asarray([len(item) for item in items], dtype=float)
        return None

    @staticmethod
    def _get_histogram(
        numbers: npt.NDArray[np.float64],
        span: tuple[float, float],
        bins: int,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
        """Return the counts, bar centres, and bar width of a histogram.

        There is one bar per value for few integers.
        """
        low, high = span
        if np.all(numbers == np.round(numbers)) and high - low < MAX_LEVELS:
            levels = np.arange(np.floor(low), np.floor(high) + 1)
            counts = np.bincount((numbers - levels[0]).astype(np.int64), minlength=len(levels))
            return counts, levels, 1.0
        low, high = (low, high) if high > low else (low - 0.5, low + 0.5)
        counts, edges = np.histogram(numbers, bins=bins, range=(low, high))
        return counts, (edges[:-1] + edges[1:]) / 2, float(edges[1] - edges[0])
