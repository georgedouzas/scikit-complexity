"""Define word elements, words of modules rewritten in parallel by string productions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Annotated

from ...errors import RewritingError
from ...specs import GeometrySpec, Letters, StateSpec, VisualSpec
from ._element import Element


class WordElement(Element):
    """Element that holds a word of modules rewritten in parallel by string productions.

    A subclass returns the `ProductionLaw`s of the word from `_get_laws`. Its states are the count of every symbol
    whose name is a word of letters, digits, and underscores, and the length `size` of the word.

    Args:
        label:
            The label of the element.

        axiom:
            The word at generation 0.

        alphabet:
            The symbols, as a string of one-character symbols or a mapping from name to number of parameters, or
            `None` to take them from the axiom and the productions.

        ignore:
            The symbols that contexts skip.

        geometry:
            The interpretation of the word as line segments, or `None`.

        max_size:
            The largest word a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` to fit the data.

        visual:
            How the word is drawn, or `None` for `default_visual_`.

    Examples:
        >>> from skcomplex.core.elements import WordElement
        >>> from skcomplex.core.laws import ProductionLaw
        >>> class AlgaeElement(WordElement):
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> AlgaeElement('algae', axiom='A').axiom
        'A'
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

    def _check(self) -> None:
        """Raise an error for an invalid geometry or a maximum size that is not a positive integer."""
        if self.geometry is not None:
            self.geometry._validated()
        if isinstance(self.max_size, bool) or not isinstance(self.max_size, int) or self.max_size < 1:
            error_msg = f'The maximum size of {self.label} must be a positive integer, but it is {self.max_size!r}.'
            raise RewritingError(error_msg)

    def _prepare(self, context: object = None) -> None:
        """Set `default_visual_`, the geometry of the word.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.
        """
        self.default_visual_ = VisualSpec('geometry')
