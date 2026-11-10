"""Fractal systems built by substitution.

The module covers the Koch curve, the Cantor set, and the Sierpiński triangle and carpet.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import PatchElement, WordElement
from ....core.components.laws import ProductionLaw, SubstitutionLaw
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import AffineSpec, GeometrySpec, ParameterSpec, PrototileSpec, VisualSpec

SHAPES = ('triangle', 'carpet')
LARGEST_RATIO = 0.5


class CantorElement(PatchElement):
    """Patch of intervals.

    Each interval is replaced by its first and its last part of the length `ratio`.

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
        >>> from skcomplex.systems.mathematics.fractals import CantorElement
        >>> CantorElement('cantor', ratio=0.25).ratio
        0.25
    """

    law_classes = (SubstitutionLaw,)

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


class KochElement(WordElement):
    """Word rewritten by the production `F → F+F--F+F` of the Koch curve.

    Args:
        label:
            The label of the word.

        axiom:
            The word at generation 0.

        geometry:
            How the word is drawn as line segments, or `None` for no geometry.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import KochElement
        >>> KochElement('koch', axiom='F').axiom
        'F'
    """

    law_classes = (ProductionLaw,)

    def _get_laws(self, s: SimpleNamespace) -> list[ProductionLaw]:
        """Return the production of the Koch curve.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The production of the word.
        """
        del s
        return [ProductionLaw('F', 'F+F--F+F')]


class SierpinskiElement(PatchElement):
    """Patch of triangles or squares.

    Each is replaced by the scaled copies of the Sierpiński triangle or carpet.

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
        >>> from skcomplex.systems.mathematics.fractals import SierpinskiElement
        >>> SierpinskiElement('sierpinski', shape='carpet').shape
        'carpet'
    """

    law_classes = (SubstitutionLaw,)

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
        """Return the prototile of a Sierpiński shape and its scaled replacing copies."""
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


class KochCurveSystem(BaseSystem):
    """The Koch curve as the L-system `F → F+F--F+F` drawn by a turtle.

    Every generation replaces each segment by four segments of a third of its length, with a bump of the turning angle
    in the middle. With the angle of 60 degrees the curve keeps its endpoints at `(0, 0)` and `(1, 0)`, its length is
    `(4/3)^n` at generation `n`, and its similarity dimension is `log 4 / log 3`, about 1.262. The system has the word
    `koch`, its symbol counts `koch.F`, and its size `koch.size`.

    Args:
        angle:
            The turning angle of the turtle in degrees.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.fractals import KochCurveSystem
        >>> simulate(KochCurveSystem(), until=3).get_values('koch.F').tolist()
        [1.0, 4.0, 16.0, 64.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'L-system', 'rewriting system', 'turtle')
    references = ('Prusinkiewicz and Lindenmayer 1990', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 5
    dt = 1

    def __init__(self, angle: float = 60.0, *, label: str = 'system') -> None:
        super().__init__(label)
        self.angle = angle

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        word = KochElement(
            'koch',
            axiom='F',
            alphabet='F+-',
            geometry=GeometrySpec(angle=self.angle, step=1.0, contraction=1 / 3),
            bounds=(-0.03, -0.05, 1.03, 0.35),
        )
        return CompositeSystem(self.label, elements=[word])


class CantorSetSystem(BaseSystem):
    """The Cantor set as the substitution of every interval by its two outer parts.

    Every generation keeps the first and the last part of each interval, each `ratio` of its length, and removes the
    middle. With the ratio of one third, generation `n` has `2^n` intervals of total length `(2/3)^n`, and the limit set
    has the similarity dimension `log 2 / log 3`, about 0.631. The system has the patch `cantor` and its number of
    intervals `cantor.size`.

    Args:
        ratio:
            The length of each kept part relative to its interval, below one half.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The ratio is not between 0 and 1/2.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.fractals import CantorSetSystem
        >>> simulate(CantorSetSystem(), until=3).get_values('cantor.size').tolist()
        [1.0, 2.0, 4.0, 8.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'substitution', 'rewriting system')
    references = ('Cantor 1883', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 8
    dt = 1

    def __init__(self, ratio: float = 1 / 3, *, label: str = 'system') -> None:
        super().__init__(label)
        self.ratio = ratio

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a ratio outside of `(0, 1/2)`, then check the parts.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.

        Raises:
            ValidationError:
                The ratio is not between 0 and 1/2.
        """
        super()._prepare(context)
        if not 0 < self.ratio < LARGEST_RATIO:
            error_msg = f'ratio = {self.ratio!r} must lie between 0 and 1/2.'
            raise ValidationError(error_msg)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        interval = PrototileSpec('interval', [(0.0, 0.0), (1.0, 0.0)])
        cantor = CantorElement('cantor', prototiles=[interval], bounds=(-0.03, -0.1, 1.03, 0.1), ratio=self.ratio)
        return CompositeSystem(self.label, elements=[cantor])


class SierpinskiSubstitutionSystem(BaseSystem):
    """The Sierpiński triangle or carpet.

    The shape is the substitution of a polygon by scaled copies of itself.

    The triangle is replaced by its three corner copies of half its side, so generation `n` has `3^n` triangles of
    total area `(3/4)^n` times the first, and the gasket has the similarity dimension `log 3 / log 2`, about 1.585. The
    square is replaced by the eight copies of a third of its side around its middle, so generation `n` has `8^n`
    squares of area `(8/9)^n`, and the carpet has the dimension `log 8 / log 3`, about 1.893. The system has the patch
    `sierpinski` and its number of polygons `sierpinski.size`.

    Args:
        shape:
            `'triangle'` or `'carpet'`.

        filled:
            Whether the geometry visual draws the polygons as a filled image instead of outlines.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.fractals import SierpinskiSubstitutionSystem
        >>> simulate(SierpinskiSubstitutionSystem(), until=2).get_values('sierpinski.size').tolist()
        [1.0, 3.0, 9.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'substitution', 'rewriting system')
    references = ('Sierpiński 1915', 'Sierpiński 1916', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 6
    dt = 1

    def __init__(self, shape: str = 'triangle', *, filled: bool = True, label: str = 'system') -> None:
        super().__init__(label)
        self.shape = shape
        self.filled = filled

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        prototile, _ = SierpinskiElement._sierpinski_tiles(self.shape)
        sierpinski = SierpinskiElement('sierpinski', shape=self.shape, prototiles=[prototile], filled=self.filled)
        return CompositeSystem(self.label, elements=[sierpinski])
