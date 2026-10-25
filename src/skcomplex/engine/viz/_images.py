"""Draw the states of a lattice group as images, with a space-time image in one dimension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.base import BaseProblem, NumericDomain
from ...core.problems import CompiledAgents, CompiledElement, Problem
from ..runtime import Frame
from ._colors import _levels_of, _phase_colors, palette_colors
from ._utils import MODULUS, PHASE
from ._visuals import BaseVisual, register_visual

MIN_HISTORY = 64
MAX_HISTORY = 256
SPACE_TIME = 'space_time'


class ImageVisual(BaseVisual):
    """Visual that logs every state of a lattice group as an image colored by value, or a tensor in 3D and more.

    A one-dimensional lattice also logs a space-time image under `<path>/space_time/<state>`, with one row per logged
    frame, the oldest at the top, and at most `history` rows. Its colors span the values of the rows it shows.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        history:
            The largest number of rows of the space-time image of a one-dimensional lattice, 0 for no space-time
            image, or `None` for as many rows as the lattice has sites, between 64 and 256.

    Examples:
        >>> from skcomplex.engine.viz import ImageVisual
        >>> ImageVisual(low=0, high=1)
        ImageVisual(states=None, low=0, high=1, history=None)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        history: int | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.history = history
        self._rows: dict[str, deque[npt.NDArray[np.float64]]] = {}

    def __repr__(self) -> str:
        """Return the options."""
        return f'ImageVisual(states={self.states!r}, low={self.low!r}, high={self.high!r}, history={self.history!r})'

    def _history_of(self, sites: int) -> int:
        """Return the number of rows of the space-time image of a one-dimensional lattice of a number of sites."""
        if self.history is not None:
            return self.history
        return int(np.clip(sites, MIN_HISTORY, MAX_HISTORY))

    def _range(self, values: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Return the values of the two ends of the color scale, the given ones or the extremes of the values."""
        low = float(values.min()) if self.low is None else float(self.low)
        high = float(values.max()) if self.high is None else float(self.high)
        return low, high

    def _log_space_time(self, recording: rr.RecordingStream, path: str, values: npt.NDArray[np.float64]) -> None:
        """Log the recent rows of a one-dimensional lattice as a space-time depth image with a color map."""
        rows = self._history_of(len(values))
        if not rows:
            return
        history = self._rows.setdefault(path, deque(maxlen=rows))
        history.append(values)
        image = np.stack(history)
        depth = rr.DepthImage(_levels_of(image, *self._range(image)), colormap='viridis', depth_range=(0.0, 255.0))
        recording.log(path, depth)

    def _log_image(
        self,
        recording: rr.RecordingStream,
        path: str,
        values: npt.NDArray[np.float64],
        *,
        cyclic: bool,
    ) -> None:
        """Log lattice values as an image colored by value or by a cyclic palette, or as a tensor in 3D and more."""
        if values.ndim > 2:  # noqa: PLR2004
            recording.log(path, rr.Tensor(values))
            return
        values = np.atleast_2d(values)
        colors = _phase_colors(values) if cyclic else palette_colors(values, *self._range(values))
        height, width = values.shape
        recording.log(
            path,
            rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8'),
        )

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a lattice group under `<path>/<state>`, and a space-time image in one dimension.

        A complex state logs its modulus under `<path>/<state>/modulus`, colored over the range, and its phase under
        `<path>/<state>/phase`, colored by a cyclic palette over `(-pi, pi]`. An angle state is colored by the same
        cyclic palette. Its space-time image shows the modulus.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.asarray(frame.lattice(name))
            path = f'{element.path}/{state}'
            magnitudes = np.abs(values) if np.iscomplexobj(values) else np.asarray(values, dtype=float)
            if magnitudes.ndim == 1:
                self._log_space_time(recording, f'{element.path}/{SPACE_TIME}/{state}', magnitudes)
            if not np.iscomplexobj(values):
                cyclic = frame.problem.states.entry(name).domain is NumericDomain.ANGLE
                self._log_image(recording, path, magnitudes, cyclic=cyclic)
                continue
            self._log_image(recording, f'{path}/{MODULUS}', magnitudes, cyclic=False)
            self._log_image(recording, f'{path}/{PHASE}', np.angle(values), cyclic=True)

    def views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return one 2D spatial view per chosen state, or a tensor view for a lattice of three or more dimensions.

        A one-dimensional lattice gets the 2D spatial views of its space-time images first, named `<state> history`.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem, with the lattice shape of every state.

        Returns:
            views:
                One view per chosen state with `<path>/<state>` as origin, two for a complex state with its modulus
                and phase as origins, after one view per state with `<path>/space_time/<state>` as origin in one
                dimension.

        Examples:
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import ImageVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('gray_scott', shape=(16, 16)))
            >>> [str(view.origin) for view in ImageVisual().views(problem.elements[0], problem)]
            ['cells/u', 'cells/v']
            >>> problem = compile_system(create('elementary_automaton'))
            >>> [str(view.origin) for view in ImageVisual().views(problem.elements[0], problem)]
            ['cells/space_time/alive', 'cells/alive']
        """
        if not isinstance(element, CompiledElement):
            return []
        histories: list[rrb.View] = []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path, shape = (
                f'{element.path}/{state}',
                (problem.lattice_shape(name) if isinstance(problem, Problem) else None) or (),
            )
            if len(shape) == 1 and self._history_of(shape[0]):
                histories.append(
                    rrb.Spatial2DView(origin=f'{element.path}/{SPACE_TIME}/{state}', name=f'{state} history'),
                )
            kind = rrb.TensorView if len(shape) > 2 else rrb.Spatial2DView  # noqa: PLR2004
            if problem.states.entry(name).is_complex:
                views += [kind(origin=f'{path}/{part}', name=f'{state} {part}') for part in (MODULUS, PHASE)]
            else:
                views.append(kind(origin=path, name=state))
        return [*histories, *views]


register_visual('image', ImageVisual)
