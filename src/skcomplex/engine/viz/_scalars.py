"""Draw the states of an element as time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..compile import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ._colors import SERIES_COLORS
from ._utils import PARTS, log_scalars
from ._visuals import BaseVisual, register_visual


def _series_index(problem: Problem, path: str, state: str) -> int:
    """Return the position of a state among the states of every element of a problem, in declaration order."""
    names = [(element.path, name) for element in problem.elements for name in element.states]
    return names.index((path, state)) if (path, state) in names else 0


def _style_series(recording: rr.RecordingStream, path: str, value: object, index: int) -> None:
    """Log static series colors for the components of a value, from the categorical palette starting at an index."""
    values = np.ravel(np.asarray(value))
    parts = [f'{path}/{part}' for part in PARTS] if np.iscomplexobj(values) else [path]
    for offset, entity in enumerate(parts):
        colors = [SERIES_COLORS[(index + offset + column) % len(SERIES_COLORS)] for column in range(values.size)]
        recording.log(entity, rr.SeriesLines(colors=colors), static=True)


class ScalarsVisual(BaseVisual):
    """Visual that logs each state component of an element as a time series.

    Every state gets static series colors from a categorical palette, by its position among the states of every element
    of the problem, so the series of one view are told apart. The lag states that keep the past values of a delayed
    state are left out.

    Args:
        components:
            The indices of the components of vector states to log, or `None` for every component.

    Examples:
        >>> from skcomplex.engine.viz import ScalarsVisual
        >>> ScalarsVisual()
        ScalarsVisual()
    """

    def __init__(self, components: Sequence[int] | None = None) -> None:
        self.components = components
        self._styled: set[str] = set()

    def __repr__(self) -> str:
        """Return the class name and the chosen components."""
        return 'ScalarsVisual()' if self.components is None else f'ScalarsVisual(components={list(self.components)!r})'

    def _value(self, value: object) -> object:
        """Return the chosen components of a vector value, or the value itself."""
        array = np.ravel(np.asarray(value))
        if self.components is None or array.size == 1:
            return value
        return array[[index for index in self.components if index < array.size]]

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every state of the element under `<path>/<state>`, a complex one under `real` and `imag` below it.

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
            path, value = f'{element.path}/{state}', self._value(frame.value(name))
            if path not in self._styled:
                _style_series(recording, path, value, _series_index(frame.problem, element.path, state))
                self._styled.add(path)
            log_scalars(recording, path, value)

    def views(self, element: CompiledElement | CompiledAgents, problem: Problem) -> list[rrb.View]:
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
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import ScalarsVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('lotka_volterra'))
            >>> view = ScalarsVisual().views(problem.elements[0], problem)[0]
            >>> type(view).__name__, str(view.origin)
            ('TimeSeriesView', 'prey')
        """
        if not getattr(element, 'states', True):
            return []
        return [rrb.TimeSeriesView(origin=element.path, name=element.path)]


register_visual('scalars', ScalarsVisual)
