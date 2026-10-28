"""Define patch elements, tiles under affine placements replaced in parallel by substitutions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, Self

from ...errors import RewritingError
from ...specs import AffineSpec, PrototileSpec, StateSpec, VisualSpec
from ._element import Element


class PatchElement(Element):
    """Element that holds tiles, each a prototile under an affine placement, replaced in parallel by substitutions.

    A subclass returns the `SubstitutionLaw`s of the patch from `_get_laws`. Its states are the count of every
    prototile and the number `size` of tiles.

    Args:
        label:
            The label of the element.

        tiles:
            Pairs of a prototile name and its placement at generation 0, or `None` for the first prototile in place.

        prototiles:
            The prototiles.

        filled:
            Whether the geometry visual draws polygons as a filled image instead of outlines.

        max_size:
            The largest number of tiles a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` for the box of the initial tiles.

        visual:
            How the patch is drawn, or `None` for `default_visual_`.

    Examples:
        >>> from skcomplex.core.elements import PatchElement
        >>> from skcomplex.specs import PrototileSpec
        >>> PatchElement('cantor', prototiles=[PrototileSpec('interval', [(0, 0), (1, 0)])]).check().filled
        False
    """

    def __init__(
        self,
        label: str,
        tiles: Annotated[Sequence[tuple[str, AffineSpec]] | None, StateSpec(domain=None)] = None,
        *,
        prototiles: Sequence[PrototileSpec] = (),
        filled: bool = False,
        max_size: int = 10_000_000,
        bounds: Sequence[float] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tiles = tiles
        self.prototiles = prototiles
        self.filled = filled
        self.max_size = max_size
        self.bounds = bounds

    def _check(self) -> Self:
        """Raise an error for no prototiles, an invalid prototile, or a tile of an unknown prototile.

        Returns:
            element:
                The element itself.
        """
        if not self.prototiles:
            error_msg = f'The patch {self.label} needs at least one prototile.'
            raise RewritingError(error_msg)
        names = {prototile.check().name for prototile in self.prototiles}
        for name, placement in self.tiles or ():
            if name not in names:
                error_msg = f'The patch {self.label} has a tile of the unknown prototile {name!r}.'
                raise RewritingError(error_msg)
            placement.check()
        return self

    def _prepare(self, context: object = None) -> Self:
        """Set `default_visual_`, the geometry of the tiles.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.

        Returns:
            element:
                The element itself.
        """
        self.default_visual_ = VisualSpec('geometry')
        return self
