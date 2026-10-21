"""Draw the traits of agents without a space as histograms and statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence, Sized
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..compile import CompiledAgents, CompiledElement, Problem
from ..model import Space
from ..solvers import AgentSnapshot
from ._visuals import SERIES_COLORS, AgentsVisual, _widened, register_visual, resolve_visual

BINS = 20
MAX_LEVELS = 50
MAX_AGENT_SERIES = 10
STATISTICS = ('mean', 'std', 'distinct')
NUMERIC_KINDS = frozenset('biuf')


def _numbers_of(values: object) -> npt.NDArray[np.float64] | None:
    """Return one number per agent: the value, the mean of a row of numbers, or the size of a collection."""
    array = np.asarray(values)
    if array.dtype.kind in NUMERIC_KINDS:
        numbers = array.astype(float)
        return numbers if numbers.ndim == 1 else numbers.reshape(len(numbers), -1).mean(axis=1)
    items = list(np.ravel(array)) if array.ndim else [array.item()]
    if items and all(isinstance(item, Sized) and not isinstance(item, str) for item in items):
        return np.asarray([len(item) for item in items], dtype=float)
    return None


def _histogram(
    numbers: npt.NDArray[np.float64],
    span: tuple[float, float],
    bins: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the counts and the left edges of a histogram, with one bar per value for few integer values."""
    low, high = span
    if np.all(numbers == np.round(numbers)) and high - low < MAX_LEVELS:
        levels = np.arange(np.floor(low), np.floor(high) + 1)
        counts = np.bincount((numbers - levels[0]).astype(np.int64), minlength=len(levels))
        return counts, levels
    low, high = (low, high) if high > low else (low - 0.5, low + 0.5)
    counts, edges = np.histogram(numbers, bins=bins, range=(low, high))
    return counts, edges[:-1]


class TraitsVisual(AgentsVisual):
    """Visual that logs the traits of agents without a space as histograms and statistics.

    For every trait of the agents, a number per agent is taken: the value of a numeric trait, the mean of a row of
    numbers such as a genome, or the size of a collection such as a vocabulary. The visual logs the histogram of these
    numbers as a `BarChart` under `<path>/<trait>/histogram`, over a range that widens to every value logged so far,
    with one bar per value when the values are few integers, and their mean and standard deviation as series under
    `<path>/<trait>/mean` and `<path>/<trait>/std`. Integer numbers, such as type labels, also log their number of
    distinct values under `<path>/<trait>/distinct`, and a group of at most ten agents logs the number of every agent
    as its own series under `<path>/<trait>/agents`. Traits of other kinds are left out.

    Args:
        color_by:
            The trait whose histogram the first view shows, or `None` for the first trait of the agents.

        color_range:
            The range of the histogram, or `None` for a range widened to every logged value.

        radius:
            Not used, kept for the options of agents visuals.

        bins:
            The number of bins of the histograms.

        shown:
            The series of the colored trait that the first view shows, among `'agents'`, `'mean'`, `'std'`, and
            `'distinct'`.

    Examples:
        >>> from skcomplex.core.viz import TraitsVisual
        >>> TraitsVisual(color_by='fitness').bins
        20
    """

    def __init__(
        self,
        color_by: str | None = None,
        color_range: tuple[float, float] | None = None,
        radius: float | None = None,
        bins: int = BINS,
        shown: Sequence[str] = ('agents', 'mean', 'std'),
    ) -> None:
        super().__init__(color_by=color_by, color_range=color_range, radius=radius)
        self.bins = bins
        self.shown = shown
        self._spans: dict[str, tuple[float, float]] = {}

    def _log_trait(self, recording: rr.RecordingStream, path: str, numbers: npt.NDArray[np.float64]) -> None:
        """Log the histogram, the mean, and the standard deviation of the numbers of one trait."""
        if path not in self._spans:
            for index, statistic in enumerate(STATISTICS):
                style = rr.SeriesLines(colors=SERIES_COLORS[index], names=statistic)
                recording.log(f'{path}/{statistic}', style, static=True)
        span = self.color_range if self.color_range is not None else _widened(self._spans.get(path), numbers)
        self._spans[path] = span
        counts, edges = _histogram(numbers, span, self.bins)
        recording.log(f'{path}/histogram', rr.BarChart(counts, abscissa=edges))
        recording.log(f'{path}/mean', rr.Scalars(float(numbers.mean())))
        recording.log(f'{path}/std', rr.Scalars(float(numbers.std())))
        if np.all(numbers == np.round(numbers)):
            recording.log(f'{path}/distinct', rr.Scalars(float(np.unique(numbers).size)))
        if numbers.size <= MAX_AGENT_SERIES:
            recording.log(f'{path}/agents', rr.Scalars(numbers))

    def log_agents(
        self,
        recording: rr.RecordingStream,
        path: str,
        snapshot: AgentSnapshot,
        space: Space | None,
    ) -> None:
        """Log the histogram and the statistics of every trait of the agents of a group at one frame.

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
        if not len(snapshot):
            return
        for trait, values in snapshot.values.items():
            numbers = _numbers_of(values)
            if numbers is not None and numbers.size:
                self._log_trait(recording, f'{path}/{trait}', numbers)

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
        """Return the statistics of one trait over time, its histogram, and the statistics of every trait.

        Args:
            element:
                The compiled agents, with their path and the names of their states.

            problem:
                The compiled problem.

        Returns:
            views:
                A time series view of the shown series of `color_by`, or of the first state, a bar chart view of its
                histogram, and a time series view of the statistics of every trait.
        """
        if not isinstance(element, CompiledAgents) or not element.states:
            return []
        path = element.path
        trait = self.color_by if self.color_by is not None else element.states[0]
        contents = [f'+ /{path}/*/mean', f'+ /{path}/*/std']
        return [
            rrb.TimeSeriesView(
                origin='/',
                contents=[f'+ /{path}/{trait}/{series}' for series in self.shown],
                name=f'{path} {trait}',
            ),
            rrb.BarChartView(origin=f'{path}/{trait}/histogram', name=f'{path} {trait} histogram'),
            rrb.TimeSeriesView(origin='/', contents=contents, name=f'{path} traits'),
        ]


def agents_visual_of(agents: CompiledAgents, problem: Problem) -> AgentsVisual:
    """Return the visual of an agent group: its declared agents visual, or the traits visual without a space.

    Args:
        agents:
            The compiled agents.

        problem:
            The compiled problem, with the space the agents live in.

    Returns:
        visual:
            The visual that logs the agents.

    Examples:
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.viz import agents_visual_of
        >>> from skcomplex.models import create
        >>> problem = compile_system(create('moran_process'))
        >>> type(agents_visual_of(problem.agents[0], problem)).__name__
        'TraitsVisual'
    """
    visual: Any = resolve_visual(agents.visual)
    if not isinstance(visual, AgentsVisual):
        visual = AgentsVisual()
    if problem.agent_space is None and type(visual) is AgentsVisual:
        return TraitsVisual(color_by=visual.color_by, color_range=visual.color_range)
    return visual


register_visual('traits', TraitsVisual)
