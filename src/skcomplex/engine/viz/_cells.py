"""Draw the states of the groups on the edges and hyperedges of a cell complex as segments, images, and polygons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.spaces import CellComplexSpace, LatticeSpace
from ...core.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ...types import Array, Indices
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseVisual

FACES = 'faces'
EDGE_RADIUS = 0.08
PLANE_AXES = 2


class CellVisual(BaseVisual):
    """Visual that logs a state of a group on the edges or hyperedges of a cell complex.

    A state on edges logs under `<path>/<state>` as `LineStrips2D` colored by value, with a cyclic palette for an angle
    state, and its oriented sums around the faces log under `<path>/<state>/faces`. A state on hyperedges logs under
    `<path>/<state>`. Values on faces log as an image of the lattice shape on a lattice complex, and as colored closed
    polygons on a network complex. A lattice of more than two dimensions logs the plane and the slice it selects.

    Args:
        state:
            The state to log, or `None` for the first state of the group.

        plane:
            The two axes of a lattice complex to draw.

        slice:
            The site every other axis of a lattice complex is fixed at.

        path:
            The entity path of the group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Raises:
        ValidationError:
            The group lives on no edges or hyperedges of a cell complex, or the plane is not two axes of the lattice.

    Examples:
        >>> from skcomplex.engine.viz import CellVisual
        >>> CellVisual('theta', plane=(0, 2)).plane
        (0, 2)
    """

    def __init__(
        self,
        state: str | None = None,
        plane: tuple[int, int] = (0, 1),
        slice: int = 0,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.state = state
        self.plane = plane
        self.slice = slice
        self.path = path
        self.recording = recording

    def _complex_of(self, element: CompiledElementSpec) -> CellComplexSpace:
        """Return the cell complex of a group, raising an error for a group on another space or part."""
        space = getattr(element.element, 'space', None)
        if not isinstance(space, CellComplexSpace) or getattr(element.element, 'on', None) not in {'edges', 'hyperedges'}:
            error_msg = f'CellVisual logs a group on the edges or hyperedges of a CellComplexSpace, not {element.path}.'
            raise ValidationError(error_msg)
        lattice = space.space
        if isinstance(lattice, LatticeSpace) and (
            len(set(self.plane)) != PLANE_AXES or not all(0 <= axis < lattice.n_dim_ for axis in self.plane)
        ):
            error_msg = f'CellVisual draws the plane {self.plane}, which is not two axes of {lattice!r}.'
            raise ValidationError(error_msg)
        return space

    def _in_slice(self, lattice: LatticeSpace) -> npt.NDArray[np.bool_]:
        """Return whether every site lies in the selected slice of a lattice."""
        coordinates = np.indices(lattice.shape_).reshape(lattice.n_dim_, -1)
        others = [axis for axis in range(lattice.n_dim_) if axis not in self.plane]
        return np.all(coordinates[others] == self.slice, axis=0)

    def _site_points(self, lattice: LatticeSpace) -> Array:
        """Return the position of every site in the plane, with the second axis of the plane running horizontally."""
        coordinates = np.indices(lattice.shape_).reshape(lattice.n_dim_, -1).astype(float)
        first, second = self.plane
        spacings = np.asarray(lattice.spacings_)
        return np.column_stack([coordinates[second] * spacings[second], coordinates[first] * spacings[first]])

    def _lattice_segments(self, space: CellComplexSpace, lattice: LatticeSpace) -> tuple[Indices, Array]:
        """Return the links in the plane and their segments, each one spacing long from its source."""
        sources = space.edges_[None][0]
        axes = space.link_axes_
        kept = np.flatnonzero(np.isin(axes, self.plane) & self._in_slice(lattice)[sources])
        points = self._site_points(lattice)
        steps = np.zeros((kept.size, PLANE_AXES))
        along_first = axes[kept] == self.plane[0]
        steps[along_first, 1] = lattice.spacings_[self.plane[0]]
        steps[~along_first, 0] = lattice.spacings_[self.plane[1]]
        starts = points[sources[kept]]
        return kept, np.stack([starts, starts + steps], axis=1)

    def _network_segments(self, space: CellComplexSpace) -> tuple[Indices, Array]:
        """Return every edge of a network complex and its segment between the layout positions of its ends."""
        layout = self._layout(space)
        nodes = np.asarray([layout[str(node)] for node in space.graph.nodes], dtype=float)
        sources, targets = space.edges_[None]
        return np.arange(sources.size), np.stack([nodes[sources], nodes[targets]], axis=1)

    def _colors_for(self, values: Array, period: float | None) -> npt.NDArray[np.uint8]:
        """Return the colors of values, from the cyclic palette for values with a period."""
        if period is None:
            return self._colors(values, None, None)
        return self._phase_colors(np.angle(np.exp(2j * np.pi * values / period)))

    def _log_segments(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: CellComplexSpace,
        values: Array,
        period: float | None,
    ) -> None:
        """Log the values of the edges as colored segments."""
        lattice = space.space
        found = self._lattice_segments(space, lattice) if isinstance(lattice, LatticeSpace) else None
        links, segments = found if found is not None else self._network_segments(space)
        colors = self._colors_for(values[links], period)
        radius = EDGE_RADIUS * (min(lattice.spacings_) if isinstance(lattice, LatticeSpace) else 0.25)
        recording.log(path, rr.LineStrips2D(segments, colors=colors, radii=radius))

    def _log_image(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: CellComplexSpace,
        lattice: LatticeSpace,
        faces: tuple[Array, float | None],
    ) -> None:
        """Log the values of the faces in the plane as an image of the shape of the lattice along the plane."""
        values, period = faces
        first, second = sorted(self.plane)
        in_plane = np.flatnonzero(np.all(space.face_planes_ == (first, second), axis=1))
        corners = np.asarray([space.hyperedges_[face][0] for face in in_plane], dtype=np.intp)
        selected = in_plane[self._in_slice(lattice)[corners]]
        coordinates = np.unravel_index(corners[self._in_slice(lattice)[corners]], lattice.shape_)
        rows, columns = coordinates[first], coordinates[second]
        height, width = lattice.shape_[first], lattice.shape_[second]
        colors = np.zeros((height, width, 3), dtype=np.uint8)
        colors[rows, columns] = self._colors_for(values[selected], period)[..., :3]
        recording.log(path, rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8'))

    def _log_polygons(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: CellComplexSpace,
        faces: tuple[Array, float | None],
    ) -> None:
        """Log the values of the faces of a network complex as colored closed polygons."""
        values, period = faces
        layout = self._layout(space)
        nodes = list(space.graph.nodes)
        strips = [
            [layout[str(nodes[node])] for node in (*cycle, cycle[0])] for cycle in space.hyperedges_
        ]
        recording.log(path, rr.LineStrips2D(strips, colors=self._colors_for(values, period), radii=EDGE_RADIUS * 0.25))

    def _log_faces(
        self,
        recording: rr.RecordingStream,
        path: str,
        space: CellComplexSpace,
        values: Array,
        period: float | None,
    ) -> None:
        """Log the values of the faces of a complex, an image on a lattice and polygons on a network."""
        if isinstance(space.space, LatticeSpace):
            self._log_image(recording, path, space, space.space, (values, period))
            return
        self._log_polygons(recording, path, space, (values, period))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the state of the group, and the oriented sums of a state on edges around the faces.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled group, with its path and the dotted names of its states.

            frame:
                The frame.
        """
        space = self._complex_of(element)
        state = self.state if self.state is not None else next(iter(element.states))
        name = element.states[state]
        values = np.ravel(np.asarray(frame.value(name), dtype=float))
        period = frame.problem.states.entry(name).period
        cycle = None if period is None else float(period)
        path = f'{element.path}/{state}'
        if getattr(element.element, 'on', None) == 'hyperedges':
            self._log_faces(recording, path, space, values, cycle)
            return
        self._log_segments(recording, path, space, values, cycle)
        self._log_faces(recording, f'{path}/{FACES}', space, space.incidences_[1] @ values, cycle)

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return one 2D spatial view of the state of the group.

        Args:
            element:
                The compiled group.

            problem:
                The compiled problem the group belongs to.

        Returns:
            views:
                The view, with `<path>/<state>` as origin.
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        self._complex_of(element)
        state = self.state if self.state is not None else next(iter(element.states))
        return [rrb.Spatial2DView(origin=f'{element.path}/{state}', name=state)]
