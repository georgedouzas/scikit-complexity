"""Draw the states of an element as time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ..problems import BaseProblemSpec, FrameSpec
from ._base import PARTS, SERIES_COLORS, BaseVisual


class ScalarsVisual(BaseVisual):
    """A visual that logs each state component of an element as a time series.

    Every state gets static series colors from a categorical palette, by its position among the states of every element
    of the problem, so the series of one view are told apart. The lag states that keep the past values of a delayed
    state are left out.

    Args:
        components:
            The indices of the components of vector states to log, or `None` for every component.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        styled_:
            The entity paths whose series colors are logged.

    Examples:
        >>> from skcomplex.core.engine.viz import ScalarsVisual
        >>> ScalarsVisual()
        ScalarsVisual()
    """

    def __init__(
        self,
        components: Sequence[int] | None = None,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.components = components
        self.path = path
        self.recording = recording

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the element, forget the styled series of earlier runs, and set the views."""
        self.styled_: set[str] = set()
        super()._prepare(context)

    def _get_value(self, value: object) -> object:
        """Return the chosen components of a vector value, or the value itself."""
        array = np.ravel(np.asarray(value))
        if self.components is None or array.size == 1:
            return value
        return array[[index for index in self.components if index < array.size]]

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log every state of the element under `<path>/<state>`.

        A complex state is logged under `real` and `imag` below it.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            if frame.problem.companions.get(name, ('',))[0] == 'lag':
                continue
            path, value = f'{element.path}/{state}', self._get_value(frame.get_value(name))
            if path not in self.styled_:
                self._log_series_style(
                    recording,
                    path,
                    value,
                    self._get_series_index(frame.problem, element.path, state),
                )
                self.styled_.add(path)
            BaseVisual._log_scalars(recording, path, value)

    def _get_views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return one time series view of the states of the element.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A time series view with the element path as origin, or no view for an element without states.

        Examples:
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.core.engine.viz import ScalarsVisual
            >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
            >>> problem = simulate(LotkaVolterraSystem(), until=0).problem
            >>> view = ScalarsVisual(path=problem.elements[0].path).prepare(problem).views_[0]
            >>> type(view).__name__, str(view.origin)
            ('TimeSeriesView', 'prey')
        """
        if not getattr(element, 'states', True):
            return []
        return [rrb.TimeSeriesView(origin=element.path, name=element.path)]

    @staticmethod
    def _get_series_index(problem: BaseProblemSpec, path: str, state: str) -> int:
        """Return the position of a state among the states of every element of a problem.

        The states are in declaration order.
        """
        names = [(element.path, name) for element in problem.elements for name in element.states]
        return names.index((path, state)) if (path, state) in names else 0

    @staticmethod
    def _log_series_style(recording: rr.RecordingStream, path: str, value: object, index: int) -> None:
        """Log static series colors for the components of a value.

        They come from the categorical palette starting at an index.
        """
        values = np.ravel(np.asarray(value))
        parts = [f'{path}/{part}' for part in PARTS] if np.iscomplexobj(values) else [path]
        for offset, entity in enumerate(parts):
            colors = [SERIES_COLORS[(index + offset + column) % len(SERIES_COLORS)] for column in range(values.size)]
            recording.log(entity, rr.SeriesLines(colors=colors), static=True)
