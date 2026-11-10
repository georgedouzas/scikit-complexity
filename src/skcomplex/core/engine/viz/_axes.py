"""Draw the principal axes of a tensor state, or the arrows of a vector state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ...types import Array
from ..problems import BaseProblemSpec, FrameSpec
from ._base import BaseVisual

IMAGE_RANK = 2
SEGMENT_FILL = 0.45


class AxisVisual(BaseVisual):
    """A visual that logs the axes of a tensor state or the arrows of a vector state.

    They are logged at every sampled site of a lattice. A tensor state logs one segment without head per sampled site
    along its principal axis, and a vector state one arrow per sampled site, under `<path>/<state>/axes`. A scalar state
    logs its image under `<path>/<state>/image`, so one view shows both. A site at the row and column of the lattice
    sits at the same pixel of the image.

    Args:
        state:
            The tensor or vector state to draw.

        scalar:
            The scalar state of the image, or `None` for no image.

        every:
            The spacing of the sampled sites along every dimension.

        path:
            The entity path of the element to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        sites_:
            The row and the column of every sampled site.

    Raises:
        ValidationError:
            The state is no tensor or vector state of a two-dimensional lattice, or the spacing is not positive.

    Examples:
        >>> from skcomplex.core.engine.viz import AxisVisual
        >>> AxisVisual('Q', scalar='rho', every=2)
        AxisVisual(every=2, scalar='rho', state='Q')
    """

    def __init__(
        self,
        state: str,
        scalar: str | None = None,
        every: int = 4,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        super().__init__(path=path, recording=recording)
        self.state = state
        self.scalar = scalar
        self.every = every

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Raise an error for a spacing of sites that is not positive.

        Args:
            context:
                The compiled problem.
        """
        if self.every < 1:
            error_msg = f'every = {self.every!r} must be a positive number of sites.'
            raise ValidationError(error_msg)
        super()._prepare(context)

    def _get_views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the 2D spatial view of the axes and the image.

        The sampled sites are found first.

        Args:
            element:
                The compiled lattice group.

            problem:
                The compiled problem, with the lattice shape of every state.

        Returns:
            views:
                One view with `<path>/<state>` as origin.

        Raises:
            ValidationError:
                The state is no tensor or vector state of a two-dimensional lattice.
        """
        name = str(getattr(element, 'states', {}).get(self.state))
        shape = problem.get_lattice_shape(name)
        if shape is None or len(shape) != IMAGE_RANK or not problem.states.get_entry(name).member_shape:
            error_msg = f'{self.path}.{self.state} is no tensor or vector state of a two-dimensional lattice.'
            raise ValidationError(error_msg)
        rows, columns = np.meshgrid(
            np.arange(0, shape[0], self.every),
            np.arange(0, shape[1], self.every),
            indexing='ij',
        )
        self.sites_ = np.stack([rows.ravel(), columns.ravel()], axis=1)
        return [rrb.Spatial2DView(origin=f'{element.path}/{self.state}', name=self.state)]

    def _get_segments(self, axes: Array) -> list[Array]:
        """Return the two ends of a segment through every sampled site.

        They are in pixels of the image.
        """
        reach = SEGMENT_FILL * self.every * axes[:, ::-1]
        centres = self.sites_[:, ::-1].astype(float)
        return list(np.stack([centres - reach, centres + reach], axis=1))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the axes or the arrows at the sampled sites, and the image of the scalar.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its variables.

            frame:
                The frame.
        """
        name = element.states[self.state]
        values = np.asarray(frame.get_lattice(name), dtype=float)[self.sites_[:, 0], self.sites_[:, 1]]
        tensor = frame.problem.states.get_entry(name).tensor
        path = f'{element.path}/{self.state}'
        if tensor is not None:
            axes, _ = tensor.get_principal_of(values)
            recording.log(f'{path}/axes', rr.LineStrips2D(self._get_segments(axes[:, :IMAGE_RANK])))
        else:
            centres = self.sites_[:, ::-1].astype(float)
            recording.log(f'{path}/axes', rr.Arrows2D(origins=centres, vectors=values[:, IMAGE_RANK - 1 :: -1]))
        if self.scalar is None:
            return
        image = np.asarray(frame.get_lattice(element.states[self.scalar]), dtype=float)
        colors = BaseVisual._get_palette_colors(image, float(image.min()), float(image.max()))
        height, width = image.shape
        recording.log(
            f'{path}/image',
            rr.Image(bytes=colors.tobytes(), width=width, height=height, color_model='RGB', datatype='U8'),
        )
