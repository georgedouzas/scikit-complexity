"""Draw a protein chain from a lattice state of coordinates or of dihedral angles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import rerun as rr

from ....core.specs import CompiledElementSpec
from ....engine.problems import BaseProblemSpec, FrameSpec
from ....engine.viz import PointsVisual
from ._backbone_geometry import BackboneGeometry
from ._structure_spec import StructureSpec

CHAIN_DIMENSIONS = 3


class BackboneVisual(PointsVisual):
    """Visual that logs the effective atoms of a chain as points in three dimensions, joined by a line strip in order.

    A state of coordinates holds one row of three values per residue. With a structure, the state holds the angles of a
    `BackboneGeometry` instead, and the visual rebuilds the chain from them at every frame.

    Args:
        position:
            The name of the lattice state of coordinates or angles.

        structure:
            The structure whose chain the angles rebuild, or `None` for a state of coordinates.

        radius:
            The radius of the points, or `None` for a share of the extent of the positions.

        states:
            The names of the other states to log as time series.

        trail:
            The number of recent positions the trail of each point holds, or 0 for no trail.

        strip:
            Whether to join the points in member order by one line strip.

        path:
            The entity path of the group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        geometry_:
            The chain the angles rebuild, or `None` for a state of coordinates.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import BackboneVisual
        >>> BackboneVisual(position='x').strip
        True
    """

    def __init__(
        self,
        position: str,
        structure: StructureSpec | None = None,
        radius: float | None = None,
        states: Sequence[str] | None = (),
        *,
        trail: int = 0,
        strip: bool = True,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        super().__init__(position, radius, states=states, trail=trail, strip=strip, path=path, recording=recording)
        self.structure = structure

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the chain that the angles rebuild, and prepare the visual."""
        self.geometry_ = None if self.structure is None else BackboneGeometry(self.structure).check().prepare()
        super()._prepare(context)

    def _read_points(self, element: CompiledElementSpec, frame: FrameSpec) -> npt.NDArray[np.float64]:
        """Return the effective atoms at a frame, rebuilt from the angles when the visual holds a structure."""
        values = np.asarray(frame.value(element.states[self.position]), dtype=float)
        if self.geometry_ is None:
            return values.reshape(-1, CHAIN_DIMENSIONS)
        return self.geometry_.run(values.ravel()).read_effective_atoms()

    def _get_dimension(self, element: CompiledElementSpec, problem: BaseProblemSpec) -> tuple[int, ...]:
        """Return the three coordinates of a point."""
        return (CHAIN_DIMENSIONS,)
