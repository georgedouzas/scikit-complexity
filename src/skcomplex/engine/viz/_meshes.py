"""Draw the states of a finite element field as colored meshes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.base import BaseProblem
from ...core.problems import CompiledAgents, CompiledElement
from ..runtime import Frame
from ._colors import _phase_colors, palette_colors
from ._utils import MODULUS, PHASE
from ._visuals import BaseVisual, register_visual


class MeshVisual(BaseVisual):
    """Visual that logs every state of a finite element field as a triangle mesh colored by value.

    Triangle meshes are drawn flat with the value as their height, and tetrahedral meshes as points.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

    Examples:
        >>> from skcomplex.engine.viz import MeshVisual
        >>> MeshVisual(low=0.0)
        MeshVisual(states=None, low=0.0, high=None)
    """

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high

    def __repr__(self) -> str:
        """Return the options."""
        return f'MeshVisual(states={self.states!r}, low={self.low!r}, high={self.high!r})'

    def _colors(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        """Return the colors of node values over the given range or the extremes of the values."""
        low = float(values.min()) if self.low is None else float(self.low)
        high = float(values.max()) if self.high is None else float(self.high)
        return palette_colors(values, low, high)

    @staticmethod
    def _log_mesh(
        recording: rr.RecordingStream,
        path: str,
        mesh: Any,  # noqa: ANN401
        values: npt.NDArray[np.float64],
        colors: npt.NDArray[np.uint8],
    ) -> None:
        """Log node values as a flat triangle mesh with the values as height, or as points of a tetrahedral mesh."""
        nodes, cells = mesh.nodes, np.asarray(mesh.cells)
        if mesh.n_dim != 2:  # noqa: PLR2004
            recording.log(path, rr.Points3D(nodes, colors=colors))
            return
        positions = np.column_stack([nodes, values])
        recording.log(path, rr.Mesh3D(vertex_positions=positions, triangle_indices=cells.T, vertex_colors=colors))

    def log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
        """Log every chosen state of a finite element field under `<path>/<state>`, a complex one as modulus and phase.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        mesh: Any = getattr(element.element, 'space', None)
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            values = np.ravel(frame.value(name))
            path = f'{element.path}/{state}'
            if not np.iscomplexobj(values):
                self._log_mesh(recording, path, mesh, values, self._colors(values))
                continue
            modulus, phase = np.abs(values), np.angle(values)
            self._log_mesh(recording, f'{path}/{MODULUS}', mesh, modulus, self._colors(modulus))
            self._log_mesh(recording, f'{path}/{PHASE}', mesh, phase, _phase_colors(phase))

    def views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return one 3D spatial view per chosen state, since meshes are drawn with the value as their height.

        Args:
            element:
                The compiled finite element field, with its path and the dotted names of its states.

            problem:
                The compiled problem the field belongs to.

        Returns:
            views:
                One 3D spatial view per chosen state, with `<path>/<state>` as origin.

        Examples:
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import MeshVisual
            >>> from skcomplex.models import create
            >>> problem = compile_system(create('heat_l_shape', refine=1))
            >>> [type(view).__name__ for view in MeshVisual().views(problem.elements[0], problem)]
            ['Spatial3DView']
        """
        if not isinstance(element, CompiledElement):
            return []
        views: list[rrb.View] = []
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path = f'{element.path}/{state}'
            if problem.states.entry(name).is_complex:
                views += [
                    rrb.Spatial3DView(origin=f'{path}/{part}', name=f'{state} {part}') for part in (MODULUS, PHASE)
                ]
            else:
                views.append(rrb.Spatial3DView(origin=path, name=state))
        return views


register_visual('mesh', MeshVisual)
