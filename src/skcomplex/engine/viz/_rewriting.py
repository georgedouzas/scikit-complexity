"""Draw the words, patches, and host graphs of rewriting systems as Rerun archetypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Literal

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...core.elements import PatchElement, WordElement
from ..compile._structures import GraphStructure, PatchStructure, Structure, WordStructure, affine_matrix, is_segment
from ..problems import BaseProblem, CompiledAgents, CompiledElement
from ..runtime import Frame
from ._colors import colors_of
from ._utils import log_scalars
from ._visuals import BaseVisual

MARGIN = 0.03
RESOLUTION = 512
FILL = np.array([94, 201, 98], dtype=np.uint8)
BACKGROUND = np.array([20, 20, 30], dtype=np.uint8)


def _bounds_of(element: CompiledElement) -> tuple[float, float, float, float] | None:
    """Return the box an element's geometry is drawn in: its declared bounds, or the box of a patch's first tiles."""
    component = element.element
    declared = getattr(component, 'bounds', None)
    if declared is not None:
        xmin, ymin, xmax, ymax = (float(item) for item in declared)
        return (xmin, ymin, xmax, ymax)
    if not isinstance(component, PatchElement) or not component.prototiles:
        return None
    tiles = component.tiles if component.tiles is not None else [(component.prototiles[0].name, None)]
    names = [prototile.name for prototile in component.prototiles]
    try:
        matrices = [np.eye(3) if placement is None else affine_matrix(placement, {}) for _, placement in tiles]
        codes = [names.index(name) for name, _ in tiles]
    except (TypeError, ValueError):
        return None
    xmin, ymin, xmax, ymax = PatchStructure(component.prototiles, codes, np.reshape(matrices, (-1, 3, 3)), 0).bounds()
    pad = MARGIN * max(xmax - xmin, ymax - ymin, 1e-12)
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def _strips(structure: Structure) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Return the line strips of a structure, with y flipped for the downward axis of Rerun, and a value per strip."""
    segments = structure.segments()
    if isinstance(structure, WordStructure):
        values = np.zeros(len(segments)) if structure.depths is None else structure.depths.astype(float)
    elif isinstance(structure, PatchStructure):
        codes = [code for code, prototile in enumerate(structure.prototiles) if is_segment(prototile)]
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


class GeometryVisual(BaseVisual):
    """Visual that logs the structure of a rewritten element at every frame, in a frame fixed across generations.

    Word segments, segment tiles, and polygon outlines are one batch of line strips, colored by branch depth or by
    prototile. A filled patch is an image of its raster instead. A host graph is logged as graph nodes and edges. The
    counts of symbols, prototiles, or labels are logged as series under `<path>/counts/<state>`.

    Args:
        filled:
            Whether to draw polygons as a filled image, or `None` to follow the patch.

        resolution:
            The side in pixels of the image of a filled patch.

    Examples:
        >>> from skcomplex.engine.viz import GeometryVisual
        >>> GeometryVisual(filled=True)
        GeometryVisual(filled=True)
    """

    def __init__(self, filled: bool | None = None, resolution: int = RESOLUTION) -> None:
        self.filled = filled
        self.resolution = resolution

    def _is_filled(self, element: CompiledElement) -> bool:
        """Check whether an element is drawn as a filled image."""
        return bool(getattr(element.element, 'filled', False) if self.filled is None else self.filled)

    def _log(self, recording: rr.RecordingStream, element: CompiledElement, frame: Frame) -> None:
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
            log_scalars(recording, f'{element.path}/counts/{state}', frame.value(name))
        structure = frame.structures.get(element.path)
        if structure is None:
            return
        if isinstance(structure, GraphStructure):
            nodes = [str(node) for node in structure.graph.nodes]
            labels = [str(label) for _, label in structure.graph.nodes(data='label', default='')]
            recording.log(element.path, rr.GraphNodes(nodes, labels=labels))
            edges = [(str(first), str(second)) for first, second in structure.graph.edges]
            kind: Literal['directed', 'undirected'] = 'directed' if structure.graph.is_directed() else 'undirected'
            recording.log(element.path, rr.GraphEdges(edges, graph_type=kind))
            return
        if isinstance(structure, PatchStructure) and self._is_filled(element):
            grid = structure.raster(self.resolution, _bounds_of(element))
            pixels = np.where(grid[..., None], FILL, BACKGROUND).astype(np.uint8)
            recording.log(element.path, rr.Image(pixels))
            return
        strips, values = _strips(structure)
        colors = colors_of(values, 0.0, max(float(values.max(initial=0.0)), 1.0))
        dimension = strips[0].shape[1] if strips else 2
        archetype = rr.LineStrips3D if dimension == 3 else rr.LineStrips2D  # noqa: PLR2004
        recording.log(element.path, archetype(strips, colors=colors))

    def _views(self, element: CompiledElement | CompiledAgents, problem: BaseProblem) -> list[rrb.View]:
        """Return the view of the geometry, with the bounds of the element, and a time series view of the counts.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                A graph view for a host graph, a 3D view for a word geometry in three dimensions, or a 2D view
                otherwise, followed by the time series view.

        Examples:
            >>> from skcomplex.core.elements import WordElement
            >>> from skcomplex.core.laws import ProductionLaw
            >>> from skcomplex.core.spaces import DiscreteTime
            >>> from skcomplex.core.specs import GeometrySpec
            >>> from skcomplex.core.system import System
            >>> from skcomplex.engine.compile import compile_system
            >>> from skcomplex.engine.viz import GeometryVisual
            >>> class KochElement(WordElement):
            ...     def _get_laws(self, namespace):
            ...         return [ProductionLaw('F', 'F+F--F+F')]
            >>> koch = KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))
            >>> problem = compile_system(System(elements=[koch], time=DiscreteTime()))
            >>> [type(view).__name__ for view in GeometryVisual().prepare(problem.elements[0], problem).views_]
            ['Spatial2DView', 'TimeSeriesView']
        """
        del problem
        path = element.path
        series = rrb.TimeSeriesView(origin=f'{path}/counts', name=f'{path} counts')
        component = element.element
        if not isinstance(component, WordElement | PatchElement):
            return [rrb.GraphView(origin=path, name=path), series]
        geometry = component.geometry if isinstance(component, WordElement) else None
        if geometry is not None and geometry.dimension == 3:  # noqa: PLR2004
            return [rrb.Spatial3DView(origin=path, name=path), series]
        bounds = _bounds_of(element) if isinstance(element, CompiledElement) else None
        if bounds is None or self._is_filled(element):  # type: ignore[arg-type]
            return [rrb.Spatial2DView(origin=path, name=path), series]
        xmin, ymin, xmax, ymax = bounds
        visual_bounds = rrb.VisualBounds2D(x_range=[xmin, xmax], y_range=[-ymax, -ymin])
        return [rrb.Spatial2DView(origin=path, name=path, visual_bounds=visual_bounds), series]
