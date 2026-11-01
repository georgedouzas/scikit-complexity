"""Patch of triangles or squares, each replaced by the scaled copies of the Sierpiński triangle or carpet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from ....errors import ValidationError
from ....specs import VisualSpec
from ._affine_spec import AffineSpec
from ._patch_element import PatchElement
from ._prototile_spec import PrototileSpec
from ._substitution_law import SubstitutionLaw

SHAPES = ('triangle', 'carpet')


class SierpinskiElement(PatchElement):
    """Patch of triangles or squares, each replaced by the scaled copies of the Sierpiński triangle or carpet.

    Args:
        label:
            The label of the patch.

        tiles:
            Pairs of a prototile name and its placement at generation 0, or `None` for the first prototile in place.

        shape:
            `'triangle'` or `'carpet'`.

        prototiles:
            The prototiles of the patch.

        filled:
            Whether the geometry visual draws polygons as a filled image instead of outlines.

        max_size:
            The largest number of tiles a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` for the box of the initial tiles.

        visual:
            How the patch is drawn, or `None` for `default_visual_`.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import SierpinskiElement
        >>> len(SierpinskiElement('sierpinski', shape='carpet')._get_laws(None)[0].pieces)
        8
    """

    def __init__(
        self,
        label: str,
        tiles: Sequence[tuple[str, AffineSpec]] | None = None,
        *,
        shape: str = 'triangle',
        prototiles: Sequence[PrototileSpec] = (),
        filled: bool = False,
        max_size: int = 10_000_000,
        bounds: Sequence[float] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(
            label,
            tiles,
            prototiles=prototiles,
            filled=filled,
            max_size=max_size,
            bounds=bounds,
            visual=visual,
        )
        self.shape = shape

    def _get_laws(self, s: SimpleNamespace) -> list[SubstitutionLaw]:
        """Return the substitution of every polygon by its scaled copies.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The substitution of the patch.

        Raises:
            ValidationError:
                The shape is unknown.
        """
        del s
        prototile, pieces = SierpinskiElement._sierpinski_tiles(self.shape)
        return [SubstitutionLaw(prototile.name, pieces)]

    @staticmethod
    def _sierpinski_tiles(shape: str) -> tuple[PrototileSpec, list[tuple[str, AffineSpec]]]:
        """Return the prototile of a Sierpiński shape and the scaled copies that replace it."""
        if shape == 'triangle':
            height = np.sqrt(3) / 2
            corners = [(0.0, 0.0), (0.5, 0.0), (0.25, height / 2)]
            pieces = [('triangle', AffineSpec(scale=0.5, translation=corner)) for corner in corners]
            return PrototileSpec('triangle', [(0.0, 0.0), (1.0, 0.0), (0.5, height)]), pieces
        if shape == 'carpet':
            cells = [(column, row) for row in range(3) for column in range(3) if (column, row) != (1, 1)]
            pieces = [('square', AffineSpec(scale=1 / 3, translation=(column / 3, row / 3))) for column, row in cells]
            return PrototileSpec('square', [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]), pieces
        error_msg = f'Unknown shape {shape!r}. Use one of {", ".join(SHAPES)}.'
        raise ValidationError(error_msg)
