"""Define the element that holds a word of symbols that productions rewrite."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Annotated

from ...errors import RewritingError
from ...specs import GeometrySpec, Letters, StateSpec, VisualSpec
from ..spaces import BaseSpace
from ._element import Element


class WordElement(Element):
    """An element that holds a word of symbols that productions rewrite in parallel.

    A word is a string of modules, symbols that may carry parameters. At each step every symbol is replaced at once by
    the word its `ProductionLaw` gives, as in an L-system, the grammar that Lindenmayer introduced to describe the
    growth of plants. A subclass returns these laws from `_get_laws`. The states of the word are the count of each
    symbol whose name is made of letters, digits, and underscores, and the length `size` of the word.

    Unlike `PatchElement`, which rewrites shapes, it rewrites a string, which `geometry` can draw as line segments.
    Choose it when a model grows by rewriting symbols, such as a branching plant or a fractal curve.

    It appears in many fields. Lindenmayer modelled the growth of algae with it in biology, the Thue-Morse sequence of
    symbolic dynamics in computer science is a word, and the Koch curve of mathematics is a word drawn as line segments.

    Args:
        label:
            The name of the word, such as `'algae'`.

        axiom:
            The word at the start, such as `'A'`.

        alphabet:
            The symbols, as a string of one-character symbols or a mapping from each name to its number of parameters,
            or `None` to read them from the axiom and the productions.

        ignore:
            The symbols that a production skips when it reads the neighbours of a symbol, such as `'+-'` for turns.

        geometry:
            How the word is drawn as line segments, a `GeometrySpec`, or `None` for no drawing.

        max_size:
            The largest word one step may produce, which guards memory.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` that the drawing shows, or `None` to fit the data.

        visual:
            How the word is drawn, such as `VisualSpec('geometry')`, or `None` for `default_visual_`.

    Examples:
        In Lindenmayer's algae `A` becomes `AB` and `B` becomes `A`, so the length of the word runs through the
        Fibonacci numbers.

        >>> from skcomplex.core.components.elements import WordElement
        >>> from skcomplex.core.components.laws import ProductionLaw
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class AlgaeElement(WordElement):
        ...     law_classes = (ProductionLaw,)
        ...
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> trajectory = simulate(CompositeSystem(elements=[AlgaeElement('algae', axiom='A')]), until=5)
        >>> trajectory.get_values('algae.size').tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    def __init__(
        self,
        label: str,
        axiom: Annotated[Letters, StateSpec(domain=None)] = '',
        *,
        alphabet: str | Mapping[str, int] | None = None,
        ignore: str | Sequence[str] = '',
        geometry: GeometrySpec | None = None,
        max_size: int = 10_000_000,
        bounds: Sequence[float] | None = None,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.axiom = axiom
        self.alphabet = alphabet
        self.ignore = ignore
        self.geometry = geometry
        self.max_size = max_size
        self.bounds = bounds

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, the geometry of the word.

        It first raises an error for an invalid geometry or a maximum size that is not positive. The maximum size must
        be a positive integer.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.
        """
        super()._prepare(context)
        if self.geometry is not None:
            self.geometry._check()
        if isinstance(self.max_size, bool) or not isinstance(self.max_size, int) or self.max_size < 1:
            error_msg = f'The maximum size of {self.label} must be a positive integer, but it is {self.max_size!r}.'
            raise RewritingError(error_msg)
        self.default_visual_ = VisualSpec('geometry')
