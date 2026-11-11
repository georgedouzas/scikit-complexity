"""Draw the states of a finite element field as colored meshes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.spaces import MeshSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ..enums import ComplexComponentKind
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseVisual

SURFACE_DIMENSION = 2


class MeshVisual(BaseVisual):
    """A visual that logs every state of a finite element field as a triangle mesh.

    The mesh is colored by value. Triangle meshes are drawn flat with the value as their height, and tetrahedral meshes
    as points.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Examples:
        >>> from skcomplex.core.engine.viz import MeshVisual
        >>> MeshVisual(low=0.0)
        MeshVisual(low=0.0)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.path = path
        self.recording = recording

    @staticmethod
    def _log_mesh(
        recording: rr.RecordingStream,
        path: str,
        mesh: MeshSpace,
        values: npt.NDArray[np.float64],
        colors: npt.NDArray[np.uint8],
    ) -> None:
        """Log node values as a flat triangle mesh or as points of a tetrahedral mesh.

        The triangle mesh has the values as its height.
        """
        nodes, cells = mesh.positions_, np.asarray(mesh.cells)
        if mesh.n_dim_ != SURFACE_DIMENSION:
            recording.log(path, rr.Points3D(nodes, colors=colors))
            return
        positions = np.column_stack([nodes, values])
        recording.log(path, rr.Mesh3D(vertex_positions=positions, triangle_indices=cells.T, vertex_colors=colors))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log every chosen state of a finite element field under `<path>/<state>`.

        A complex state is logged as modulus and phase.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.

        Raises:
            ValidationError:
                The element does not lie on a mesh space.
        """
        mesh = getattr(element.element, 'space', None)
        if not isinstance(mesh, MeshSpace):
            error_msg = f'The element {element.path} does not lie on a mesh space. Draw it with another visual.'
            raise ValidationError(error_msg)
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.ravel(frame.get_value(name))
            path = f'{element.path}/{state}'
            if not np.iscomplexobj(values):
                self._log_mesh(recording, path, mesh, values, self._get_colors(values, self.low, self.high))
                continue
            modulus, phase = np.abs(values), np.angle(values)
            self._log_mesh(
                recording,
                f'{path}/{ComplexComponentKind.MODULUS}',
                mesh,
                modulus,
                self._get_colors(modulus, self.low, self.high),
            )
            self._log_mesh(
                recording,
                f'{path}/{ComplexComponentKind.PHASE}',
                mesh,
                phase,
                BaseVisual._get_phase_colors(phase),
            )

    def _get_views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return one 3D spatial view per chosen state.

        Meshes are drawn with the value as their height.

        Args:
            element:
                The compiled finite element field, with its path and the dotted names of its states.

            problem:
                The compiled problem the field belongs to.

        Returns:
            views:
                One 3D spatial view per chosen state, with `<path>/<state>` as origin.

        Examples:
            >>> from skcomplex.core.engine.runtime import simulate
            >>> from skcomplex.core.engine.viz import MeshVisual
            >>> from skcomplex.systems.physics.heat import HeatLShapeSystem
            >>> problem = simulate(HeatLShapeSystem(refine=1), until=0).problem
            >>> views = MeshVisual(path=problem.elements[0].path).prepare(problem).views_
            >>> [type(view).__name__ for view in views]
            ['Spatial3DView']
        """
        if not isinstance(element, CompiledElementSpec):
            return []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path = f'{element.path}/{state}'
            if problem.states.get_entry(name).is_complex:
                views += [
                    rrb.Spatial3DView(origin=f'{path}/{part}', name=f'{state} {part}')
                    for part in (ComplexComponentKind.MODULUS, ComplexComponentKind.PHASE)
                ]
            else:
                views.append(rrb.Spatial3DView(origin=path, name=state))
        return views
