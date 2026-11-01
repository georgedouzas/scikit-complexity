"""Patch of intervals, each replaced by its first and its last part of the length `ratio`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....specs import ParameterSpec, VisualSpec
from ._affine_spec import AffineSpec
from ._patch_element import PatchElement
from ._prototile_spec import PrototileSpec
from ._substitution_law import SubstitutionLaw


class CantorElement(PatchElement):
    """Patch of intervals, each replaced by its first and its last part of the length `ratio`.

    Args:
        label:
            The label of the patch.

        tiles:
            Pairs of a prototile name and its placement at generation 0, or `None` for the first prototile in place.

        ratio:
            The length of each kept part relative to its interval.

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
        >>> from skcomplex.models.mathematics.fractals import CantorElement
        >>> CantorElement('cantor', ratio=0.25).ratio
        0.25
    """

    def __init__(
        self,
        label: str,
        tiles: Sequence[tuple[str, AffineSpec]] | None = None,
        *,
        ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1 / 3,
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
        self.ratio = ratio

    def _get_laws(self, s: SimpleNamespace) -> list[SubstitutionLaw]:
        """Return the substitution of every interval by its two outer parts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The substitution of the patch.
        """
        pieces = [
            ('interval', AffineSpec(scale=s.ratio)),
            ('interval', AffineSpec(scale=s.ratio, translation=(1 - s.ratio, 0))),
        ]
        return [SubstitutionLaw('interval', pieces)]
