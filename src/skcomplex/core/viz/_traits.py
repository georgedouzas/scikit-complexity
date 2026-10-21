"""Draw the traits of agents without a space as histograms and statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sized
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ..compile import CompiledAgents, CompiledElement, Problem
from ..model import Space
from ..solvers import AgentSnapshot
from ._visuals import AgentsVisual, _widened, register_visual, resolve_visual

BINS = 20
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


class TraitsVisual(AgentsVisual):
    """Visual that logs the traits of agents without a space as histograms and statistics.

    For every trait of the agents, a number per agent is taken: the value of a numeric trait, the mean of a row of
    numbers such as a genome, or the size of a collection such as a vocabulary. The visual logs the histogram of these
    numbers as a `BarChart` under `<path>/<trait>/histogram`, over a range that widens to every value logged so far, and
    their mean and standard deviation as series under `<path>/<trait>/mean` and `<path>/<trait>/std`. Traits of other
    kinds are left out.

    Args:
        color_by:
            The trait whose histogram the first view shows, or `None` for the first trait of the agents.

        color_range:
            The range of the histogram, or `None` for a range widened to every logged value.

        radius:
            Not used, kept for the options of agents visuals.

        bins:
            The number of bins of the histograms.

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
    ) -> None:
        super().__init__(color_by=color_by, color_range=color_range, radius=radius)
        self.bins = bins
        self._spans: dict[str, tuple[float, float]] = {}

    def _log_trait(self, recording: rr.RecordingStream, path: str, numbers: npt.NDArray[np.float64]) -> None:
        """Log the histogram, the mean, and the standard deviation of the numbers of one trait."""
        span = self.color_range if self.color_range is not None else _widened(self._spans.get(path), numbers)
        self._spans[path] = span
        low, high = span if span[1] > span[0] else (span[0] - 0.5, span[0] + 0.5)
        counts, edges = np.histogram(numbers, bins=self.bins, range=(low, high))
        recording.log(f'{path}/histogram', rr.BarChart(counts, abscissa=edges[:-1]))
        recording.log(f'{path}/mean', rr.Scalars(float(numbers.mean())))
        recording.log(f'{path}/std', rr.Scalars(float(numbers.std())))

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
        """Return a bar chart view of the histogram of one trait and a time series view of every trait's statistics.

        Args:
            element:
                The compiled agents, with their path and the names of their states.

            problem:
                The compiled problem.

        Returns:
            views:
                A bar chart view of the histogram of `color_by`, or of the first state, and a time series view of the
                means and standard deviations.
        """
        if not isinstance(element, CompiledAgents) or not element.states:
            return []
        path = element.path
        trait = self.color_by if self.color_by is not None else element.states[0]
        contents = [f'+ /{path}/*/mean', f'+ /{path}/*/std']
        return [
            rrb.BarChartView(origin=f'{path}/{trait}/histogram', name=f'{path} {trait}'),
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
