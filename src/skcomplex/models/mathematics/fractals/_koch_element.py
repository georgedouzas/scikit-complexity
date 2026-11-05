"""Word rewritten by the production `F → F+F--F+F` of the Koch curve."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from ....core.elements import WordElement
from ....core.laws import ProductionLaw


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
        >>> from skcomplex.models.mathematics.fractals import KochElement
        >>> KochElement('koch', axiom='F')._get_laws(None)[0].successor
        'F+F--F+F'
    """

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
