"""Draw the states of a lattice group as images, with a space-time image in one dimension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.spaces import PhaseSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ..problems import BaseProblemSpec, FrameSpec
from ._base import _MODULUS, _PHASE, BaseVisual

_MIN_HISTORY = 64
_MAX_HISTORY = 256
_IMAGE_RANK = 2
_SPACE_TIME = 'space_time'


class ImageVisual(BaseVisual):
    """Visual that logs every state of a lattice group as an image colored by value, or a tensor in 3D and more.

    A one-dimensional lattice also logs a space-time image under `<path>/space_time/<state>`, with one row per logged
    frame, the oldest at the top, and at most `history` rows. Its colors span the values of the rows it shows.

    A state on a phase space of more than two axes can log a plane through `slices`. Its moments over the class axes log
    under `<path>/<state>/moment_<order>`, as a bar series over a one-dimensional position lattice and as an image over
    a two-dimensional one.

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

        slices:
            The index that every axis but two is fixed at, by the name of a class axis or the index of an axis, or
            `None` to log the whole state.

        moments:
            The orders of the moments over the class axes of a phase space to log, or `None` for none.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        rows_:
            The recent rows of the space-time image of every state, by entity path.

    Raises:
        ValidationError:
            `slices` or `moments` are given for a group that is not on a phase space, or `slices` does not leave two
            axes.

    Examples:
        >>> from skcomplex.core.engine.viz import ImageVisual
        >>> ImageVisual(low=0, high=1)
        ImageVisual(high=1, low=0)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        history: int | None = None,
        *,
        slices: Mapping[str | int, int] | None = None,
        moments: Sequence[int] | None = None,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.history = history
        self.slices = slices
        self.moments = moments
        self.path = path
        self.recording = recording

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the element, forget the rows of earlier runs, and set the views."""
        self.rows_: dict[str, deque[npt.NDArray[np.float64]]] = {}
        self.plane_: dict[str | int, int] = {}
        super()._prepare(context)

    def _phase_space_of(self, element: CompiledElementSpec) -> PhaseSpace:
        """Return the phase space of a group, raising an error for a group on another space."""
        space = getattr(element.element, 'space', None)
        if not isinstance(space, PhaseSpace):
            error_msg = f'ImageVisual logs slices and moments of a phase space, but {element.path} lives on {space!r}.'
            raise ValidationError(error_msg)
        return space

    def _plane_of(self, space: PhaseSpace) -> dict[str | int, int]:
        """Return the fixed index of every axis of a plane through a phase space, by the index of the axis."""
        plane = {
            space.class_axes_[axis] if isinstance(axis, str) and axis in space.class_axes_ else axis: index
            for axis, index in (self.slices or {}).items()
        }
        if len(plane) != space.n_dim_ - _IMAGE_RANK or not all(isinstance(axis, int) for axis in plane):
            error_msg = (
                f'ImageVisual fixes the axes {list(plane)} of a phase space of {space.n_dim_} axes. Fix every axis but '
                'two, each by the name of a class axis or the index of an axis.'
            )
            raise ValidationError(error_msg)
        return plane

    def _sliced(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the plane of the values of a state, or the whole state without slices."""
        index: list[int | slice] = [slice(None)] * values.ndim
        for axis, position in self.plane_.items():
            index[int(axis)] = position
        return values[tuple(index)]

    def _moment_of(self, values: npt.NDArray[np.float64], space: PhaseSpace, order: int) -> npt.NDArray[np.float64]:
        """Return a moment of the values of a state over the class axes of a phase space, a field over position."""
        axes = tuple(space.class_axes_.values())
        grids = np.meshgrid(*space.coordinates_.values(), indexing='ij')
        weights = sum(grid**2 for grid in grids) ** (order / 2) * np.prod([space.spacings_[axis] for axis in axes])
        return np.tensordot(values, weights, axes=(axes, tuple(range(len(axes)))))

    def _log_moments(
        self,
        recording: rr.RecordingStream,
        element: CompiledElementSpec,
        path: str,
        values: npt.NDArray[np.float64],
    ) -> None:
        """Log the moments of a state over the class axes, as a bar series or an image over position."""
        space = self._phase_space_of(element)
        for order in self.moments or ():
            moment = self._moment_of(values, space, order)
            if moment.ndim == 1:
                recording.log(f'{path}/moment_{order}', rr.BarChart(moment))
            else:
                self._log_image(recording, f'{path}/moment_{order}', moment, cyclic=False)

    def _history_of(self, sites: int) -> int:
        """Return the number of rows of the space-time image of a one-dimensional lattice of a number of sites."""
        if self.history is not None:
            return self.history
        return int(np.clip(sites, _MIN_HISTORY, _MAX_HISTORY))

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
        history = self.rows_.setdefault(path, deque(maxlen=rows))
        history.append(values)
        image = np.stack(history)
        depth = rr.DepthImage(self._levels_of(image, *self._range(image)), colormap='viridis', depth_range=(0.0, 255.0))
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
        if values.ndim > _IMAGE_RANK:
            recording.log(path, rr.Tensor(values))
            return
        values = np.atleast_2d(values)
        colors = (
            BaseVisual._phase_colors(values) if cyclic else BaseVisual._palette_colors(values, *self._range(values))
        )
        height, width = values.shape
        recording.log(
            path,
            rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8'),
        )

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
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
            if (self.states is not None and state not in self.states) or frame.problem.states.entry(name).member_shape:
                continue
            values = np.asarray(frame.lattice(name))
            path = f'{element.path}/{state}'
            if self.moments:
                self._log_moments(recording, element, path, np.asarray(values, dtype=float))
            if self.plane_:
                values = self._sliced(np.asarray(values, dtype=float))
            magnitudes = np.abs(values) if np.iscomplexobj(values) else np.asarray(values, dtype=float)
            if magnitudes.ndim == 1:
                self._log_space_time(recording, f'{element.path}/{_SPACE_TIME}/{state}', magnitudes)
            if not np.iscomplexobj(values):
                cyclic = frame.problem.states.entry(name).period is not None
                self._log_image(recording, path, magnitudes, cyclic=cyclic)
                continue
            self._log_image(recording, f'{path}/{_MODULUS}', magnitudes, cyclic=False)
            self._log_image(recording, f'{path}/{_PHASE}', np.angle(np.asarray(values, dtype=complex)), cyclic=True)

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
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
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.core.engine.viz import ImageVisual
            >>> from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
            >>> problem = simulate(GrayScottSystem(shape=(16, 16)), until=0).problem
            >>> views = ImageVisual(path=problem.elements[0].path).prepare(problem).views_
            >>> [str(view.origin) for view in views]
            ['cells/u', 'cells/v']
            >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> problem = simulate(ElementaryAutomatonSystem(), until=0).problem
            >>> views = ImageVisual(path=problem.elements[0].path).prepare(problem).views_
            >>> [str(view.origin) for view in views]
            ['cells/space_time/alive', 'cells/alive']
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        if self.slices is not None:
            self.plane_ = self._plane_of(self._phase_space_of(element))
        histories: list[rrb.View] = []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if (self.states is not None and state not in self.states) or problem.states.entry(name).member_shape:
                continue
            path, shape = (
                f'{element.path}/{state}',
                problem.lattice_shape(name) or (),
            )
            if len(shape) == 1 and self._history_of(shape[0]):
                histories.append(
                    rrb.Spatial2DView(origin=f'{element.path}/{_SPACE_TIME}/{state}', name=f'{state} history'),
                )
            kind = rrb.TensorView if len(shape) - len(self.plane_) > _IMAGE_RANK else rrb.Spatial2DView
            views += self._moment_views(element, path)
            if problem.states.entry(name).is_complex:
                views += [kind(origin=f'{path}/{part}', name=f'{state} {part}') for part in (_MODULUS, _PHASE)]
            else:
                views.append(kind(origin=path, name=state))
        return [*histories, *views]

    def _moment_views(self, element: CompiledElementSpec, path: str) -> list[rrb.View]:
        """Return the view of every logged moment, a bar chart over one position axis and else a 2D spatial view."""
        if not self.moments:
            return []
        rank = len(self._phase_space_of(element).position_axes_)
        kind = rrb.BarChartView if rank == 1 else rrb.Spatial2DView
        return [kind(origin=f'{path}/moment_{order}', name=f'moment {order}') for order in self.moments]

    @staticmethod
    def _levels_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
        """Return the values as levels from 1 to 255 of a color map, with 0 left for no data."""
        if high <= low:
            return np.ones(values.shape, dtype=np.uint8)
        return (1 + np.clip((values - low) * (254.0 / (high - low)), 0.0, 254.0)).astype(np.uint8)
