"""Define the element that holds tiles that substitutions replace in parallel."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace
from ....core.engine.compile import BaseCompiler
from ....core.errors import RewritingError
from ....core.specs import StateSpec, VisualSpec
from ._affine_spec import AffineSpec
from . import _rewriting_compiler
from ._geometry_visual import GeometryVisual
from ._prototile_spec import PrototileSpec


class PatchElement(Element):
    """An element that holds tiles that substitutions replace in parallel.

    A tile is a copy of a prototile, a basic shape, under an affine placement, a map that scales, rotates, and moves it.
    At each step every tile is replaced at once by the tiles that its `SubstitutionLaw` gives. A subclass returns these
    laws from `_get_laws`. The states of the patch are the count of each prototile and the number `size` of tiles.

    Unlike `WordElement`, which rewrites a string of symbols, it rewrites shapes in the plane. Choose it when a model
    grows a geometric pattern by substitution, such as a fractal or a tiling that never repeats.

    It appears in mathematics and physics. The Cantor set and the Sierpinski triangle are built by substitution, and
    Penrose tilings, which never repeat, model the atomic order of quasicrystals.

    Args:
        label:
            The name of the patch, such as `'cantor'`.

        tiles:
            Each tile at the start, as a pair of a prototile name and its placement, or `None` for the first prototile
            in place.

        prototiles:
            The basic shapes, each a `PrototileSpec` with a name and the corners of its polygon, such as
            `PrototileSpec('interval', [(0, 0), (1, 0)])`.

        filled:
            Whether the drawing fills the polygons as an image instead of drawing their outlines.

        max_size:
            The largest number of tiles one step may produce, which guards memory.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` that the drawing shows, or `None` for the box of the initial tiles.

    Examples:
        Each step keeps the outer thirds of every interval, which builds the Cantor set.

        >>> from skcomplex.systems.general.rewriting import PatchElement
        >>> from skcomplex.systems.general.rewriting import SubstitutionLaw
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.general.rewriting import AffineSpec
        >>> from skcomplex.systems.general.rewriting import PrototileSpec
        >>> class CantorElement(PatchElement):
        ...     law_classes = (SubstitutionLaw,)
        ...
        ...     def _get_laws(self, namespace):
        ...         thirds = [AffineSpec(scale=1 / 3), AffineSpec(scale=1 / 3, translation=(2 / 3, 0))]
        ...         return [SubstitutionLaw('interval', [('interval', third) for third in thirds])]
        >>> cantor = CantorElement('cantor', prototiles=[PrototileSpec('interval', [(0, 0), (1, 0)])])
        >>> simulate(CompositeSystem(elements=[cantor]), until=3).get_values('cantor.size').tolist()
        [1.0, 2.0, 4.0, 8.0]
    """

    visual = VisualSpec(GeometryVisual)

    def __init__(
        self,
        label: str,
        tiles: Annotated[Sequence[tuple[str, AffineSpec]] | None, StateSpec(domain=None)] = None,
        *,
        prototiles: Sequence[PrototileSpec] = (),
        filled: bool = False,
        max_size: int = 10_000_000,
        bounds: Sequence[float] | None = None,
    ) -> None:
        super().__init__(label)
        self.tiles = tiles
        self.prototiles = prototiles
        self.filled = filled
        self.max_size = max_size
        self.bounds = bounds

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Check the prototiles of the patch.

        It first raises an error for no prototiles, an invalid prototile, or an unknown one. An unknown prototile is one
        a tile names.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.
        """
        super()._prepare(context)
        if not self.prototiles:
            error_msg = f'The patch {self.label} needs at least one prototile.'
            raise RewritingError(error_msg)
        names = {prototile.name for prototile in self.prototiles}
        for name, placement in self.tiles or ():
            if name not in names:
                error_msg = f'The patch {self.label} has a tile of the unknown prototile {name!r}.'
                raise RewritingError(error_msg)
            placement._check()

    def _get_compiler(self) -> BaseCompiler:
        """Return the rewriting compiler, which compiles every patch of a system to a kernel."""
        return _rewriting_compiler.RewritingCompiler()
