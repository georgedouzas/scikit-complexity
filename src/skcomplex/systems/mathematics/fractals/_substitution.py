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

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.rewriting import (
    AffineSpec,
    GeometrySpec,
    PatchElement,
    ProductionLaw,
    PrototileSpec,
    RewritingInteraction,
    SubstitutionLaw,
    WordElement,
)

SHAPES = ('triangle', 'carpet')
LARGEST_RATIO = 0.5


class CantorInteraction(Interaction):
    """Removal of the middle of every interval of a patch.

    Each interval is replaced by its first and its last part of the length `ratio`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the patch of intervals.

        ratio:
            The length of each kept part relative to its interval.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import CantorInteraction
        >>> CantorInteraction('removal', target='cantor', ratio=0.25).ratio
        0.25
    """

    law_classes = (SubstitutionLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        ratio: Annotated[float, RealSet.open(0, oo)] = 1 / 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.ratio = ratio

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the substitution of every interval by its two outer parts."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            pieces = [
                ('interval', AffineSpec(scale=context.ratio)),
                ('interval', AffineSpec(scale=context.ratio, translation=(1 - context.ratio, 0))),
            ]
            self.laws_: list[BaseComponent] = [SubstitutionLaw('interval', pieces)]


class SierpinskiInteraction(Interaction):
    """Replacement of every triangle or square of a patch by the scaled copies of the Sierpiński triangle or carpet.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the patch.

        shape:
            `'triangle'` or `'carpet'`.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import SierpinskiInteraction
        >>> SierpinskiInteraction('copies', target='sierpinski', shape='carpet').shape
        'carpet'
    """

    law_classes = (SubstitutionLaw,)

    def __init__(self, label: str, *, target: Role, shape: str = 'triangle') -> None:
        super().__init__(label)
        self.target = target
        self.shape = shape

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the substitution of every polygon by its scaled copies.

        Raises:
            ValidationError:
                The shape is unknown.
        """
        super()._prepare(context)
        prototile, pieces = self._get_sierpinski_tiles(self.shape)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [SubstitutionLaw(prototile.name, pieces)]

    @staticmethod
    def _get_sierpinski_tiles(shape: str) -> tuple[PrototileSpec, list[tuple[str, AffineSpec]]]:
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import KochCurveSystem
        >>> KochCurveSystem().prepare().simulate(until=3).get_values('koch.F').tolist()
        [1.0, 4.0, 16.0, 64.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'L-system', 'rewriting system', 'turtle')
    references = ('Prusinkiewicz and Lindenmayer 1990', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 5
    dt = 1

    def __init__(
        self,
        angle: float = 60.0,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.angle = angle

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        bend = RewritingInteraction('bend', target='koch', laws=[ProductionLaw('F', 'F+F--F+F')])
        word = WordElement(
            'koch',
            axiom='F',
            alphabet='F+-',
            geometry=GeometrySpec(angle=self.angle, step=1.0, contraction=1 / 3),
            bounds=(-0.03, -0.05, 1.03, 0.35),
        )
        self.system_ = System(self.label, elements=[word], interactions=[bend])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The ratio is not between 0 and 1/2.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import CantorSetSystem
        >>> CantorSetSystem().prepare().simulate(until=3).get_values('cantor.size').tolist()
        [1.0, 2.0, 4.0, 8.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'substitution', 'rewriting system')
    references = ('Cantor 1883', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 8
    dt = 1

    def __init__(
        self,
        ratio: float = 1 / 3,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.ratio = ratio

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The ratio is not between 0 and 1/2.
        """
        super()._prepare(context)
        if not 0 < self.ratio < LARGEST_RATIO:
            error_msg = f'ratio = {self.ratio!r} must lie between 0 and 1/2.'
            raise ValidationError(error_msg)
        interval = PrototileSpec('interval', [(0.0, 0.0), (1.0, 0.0)])
        cantor = PatchElement('cantor', prototiles=[interval], bounds=(-0.03, -0.1, 1.03, 0.1))
        removal = CantorInteraction('removal', target='cantor', ratio=self.ratio)
        self.system_ = System(self.label, elements=[cantor], interactions=[removal])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import SierpinskiSubstitutionSystem
        >>> SierpinskiSubstitutionSystem().prepare().simulate(until=2).get_values('sierpinski.size').tolist()
        [1.0, 3.0, 9.0]
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'substitution', 'rewriting system')
    references = ('Sierpiński 1915', 'Sierpiński 1916', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023')
    until = 6
    dt = 1

    def __init__(
        self,
        shape: str = 'triangle',
        *,
        filled: bool = True,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.filled = filled

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        prototile, _ = SierpinskiInteraction._get_sierpinski_tiles(self.shape)
        sierpinski = PatchElement('sierpinski', prototiles=[prototile], filled=self.filled)
        copies = SierpinskiInteraction('copies', target='sierpinski', shape=self.shape)
        self.system_ = System(self.label, elements=[sierpinski], interactions=[copies])
