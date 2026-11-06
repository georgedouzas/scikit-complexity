"""Word rewritten by the Thue-Morse substitution `0 → 01`, `1 → 10`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from ....core.components.elements import WordElement
from ....core.components.laws import ProductionLaw


class MorseElement(WordElement):
    """Word rewritten by the Thue-Morse substitution `0 → 01`, `1 → 10`.

    From the axiom `0`, generation `n` is the word `a_n = T(0) T(1) ... T(2^n - 1)` of the Thue-Morse sequence, whose
    term `T(k)` is the parity of the number of ones in the binary expansion of `k`.

    Args:
        label:
            The label of the word.

        axiom:
            The word at generation 0.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import MorseElement
        >>> [law.successor for law in MorseElement('morse', axiom='0')._get_laws(None)]
        ['01', '10']
    """

    def _get_laws(self, s: SimpleNamespace) -> list[ProductionLaw]:
        """Return the productions of the Thue-Morse substitution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The productions of the word.
        """
        del s
        return [ProductionLaw('0', '01'), ProductionLaw('1', '10')]
