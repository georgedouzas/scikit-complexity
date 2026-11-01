"""Draw the words and patches of rewriting systems as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ....core.specs import CompiledAgentsSpec, CompiledElementSpec
from ....engine.problems import BaseProblemSpec, FrameSpec
from ....engine.viz import BaseVisual
from ....specs import BaseStructureSpec
from ._patch_structure_spec import PatchStructureSpec
from ._word_structure_spec import WordStructureSpec

MARGIN = 0.03
SPACE_DIMENSION = 3
_RESOLUTION = 512
_FILL = np.array([94, 201, 98], dtype=np.uint8)
_BACKGROUND = np.array([20, 20, 30], dtype=np.uint8)


class GeometryVisual(BaseVisual):
    """Visual that logs the structure of a rewritten element at every frame, in a frame fixed across generations.

    Word segments, segment tiles, and polygon outlines are one batch of line strips, colored by branch depth or by
    prototile. A filled patch is an image of its raster instead. The counts of symbols or prototiles are logged as
    series under `<path>/counts/<state>`.

    Args:
        filled:
            Whether to draw polygons as a filled image, or `None` to follow the patch.

        resolution:
            The side in pixels of the image of a filled patch.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import GeometryVisual
        >>> GeometryVisual(filled=True)
        GeometryVisual(filled=True)
    """

    def __init__(
        self,
        filled: bool | None = None,
        resolution: int = _RESOLUTION,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.filled = filled
        self.resolution = resolution
        self.path = path
        self.recording = recording

    def _is_filled(self, element: CompiledElementSpec) -> bool:
        """Check whether an element is drawn as a filled image."""
        return bool(getattr(element.element, 'filled', False) if self.filled is None else self.filled)

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the structure of the element at one frame and its counts.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path and the dotted names of its counts.

            frame:
                The frame.
        """
        for state, name in element.states.items():
            BaseVisual._log_scalars(recording, f'{element.path}/counts/{state}', frame.value(name))
        structure = frame.structures.get(element.path)
        if not isinstance(structure, BaseStructureSpec):
            return
        if isinstance(structure, PatchStructureSpec) and self._is_filled(element):
            grid = structure.raster(self.resolution, self._bounds_of(element))
            pixels = np.where(grid[..., None], _FILL, _BACKGROUND).astype(np.uint8)
            recording.log(element.path, rr.Image(pixels))
            return
        strips, values = self._strips(structure)
        colors = BaseVisual._colors_of(values, 0.0, max(float(values.max(initial=0.0)), 1.0))
        dimension = strips[0].shape[1] if strips else 2
        archetype = rr.LineStrips3D if dimension == SPACE_DIMENSION else rr.LineStrips2D
        recording.log(element.path, archetype(strips, colors=colors))

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the view of the geometry, with the bounds of the element, and a time series view of the counts.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A 3D view for a word geometry in three dimensions, or a 2D view
                otherwise, followed by the time series view.

        Examples:
            >>> from skcomplex.models.mathematics.fractals import WordElement
            >>> from skcomplex.models.mathematics.fractals import ProductionLaw
            >>> from skcomplex.core.times import DiscreteTime
            >>> from skcomplex.models.mathematics.fractals import GeometrySpec
            >>> from skcomplex.core.system import System
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.models.mathematics.fractals import GeometryVisual
            >>> class KochElement(WordElement):
            ...     def _get_laws(self, namespace):
            ...         return [ProductionLaw('F', 'F+F--F+F')]
            >>> koch = KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))
            >>> from skcomplex.models.mathematics.fractals import RewritingCompiler
            >>> problem = compile_system(System(elements=[koch], time=DiscreteTime()), compiler=RewritingCompiler())
            >>> views = GeometryVisual(path=problem.elements[0].path).check().prepare(problem).views_
            >>> [type(view).__name__ for view in views]
            ['Spatial2DView', 'TimeSeriesView']
        """
        del problem
        path = element.path
        series = rrb.TimeSeriesView(origin=f'{path}/counts', name=f'{path} counts')
        component = element.element
        geometry = getattr(component, 'geometry', None)
        if geometry is not None and geometry.dimension == SPACE_DIMENSION:
            return [rrb.Spatial3DView(origin=path, name=path), series]
        bounds = self._bounds_of(element) if isinstance(element, CompiledElementSpec) else None
        if bounds is None or self._is_filled(element):
            return [rrb.Spatial2DView(origin=path, name=path), series]
        xmin, ymin, xmax, ymax = bounds
        visual_bounds = rrb.VisualBounds2D(x_range=[xmin, xmax], y_range=[-ymax, -ymin])
        return [rrb.Spatial2DView(origin=path, name=path, visual_bounds=visual_bounds), series]

    @staticmethod
    def _bounds_of(element: CompiledElementSpec) -> tuple[float, float, float, float] | None:
        """Return the box an element's geometry is drawn in, its declared bounds or the box of a patch's first tiles."""
        component = element.element
        declared = getattr(component, 'bounds', None)
        if declared is not None:
            xmin, ymin, xmax, ymax = (float(item) for item in declared)
            return (xmin, ymin, xmax, ymax)
        if not getattr(component, 'prototiles', None):
            return None
        tiles = component.tiles if component.tiles is not None else [(component.prototiles[0].name, None)]
        names = [prototile.name for prototile in component.prototiles]
        try:
            matrices = [np.eye(3) if placement is None else placement.matrix_at({}) for _, placement in tiles]
            codes = [names.index(name) for name, _ in tiles]
        except (TypeError, ValueError):
            return None
        xmin, ymin, xmax, ymax = PatchStructureSpec(
            tuple(component.prototiles),
            BaseStructureSpec._frozen(codes, np.int64),
            BaseStructureSpec._frozen(np.reshape(matrices, (-1, 3, 3)), np.float64),
            0,
        ).bounds()
        pad = MARGIN * max(xmax - xmin, ymax - ymin, 1e-12)
        return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

    @staticmethod
    def _strips(structure: BaseStructureSpec) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the line strips of a structure, with y flipped for Rerun's downward axis, and a value per strip."""
        segments = structure.segments()
        if isinstance(structure, WordStructureSpec):
            values = np.zeros(len(segments)) if structure.depths is None else structure.depths.astype(float)
        elif isinstance(structure, PatchStructureSpec):
            codes = [code for code, prototile in enumerate(structure.prototiles) if prototile.is_segment]
            values = np.concatenate(
                [np.zeros(0), *(np.full(np.count_nonzero(structure.codes == code), float(code)) for code in codes)],
            )
        else:
            values = np.zeros(len(segments))
        strips = list(segments.copy())
        for index, polygons in enumerate(structure.polygons()):
            strips += [np.concatenate([polygon, polygon[:1]]) for polygon in polygons]
            values = np.concatenate([values, np.full(len(polygons), float(index))])
        for strip in strips:
            strip[:, 1] *= -1
        return strips, values
