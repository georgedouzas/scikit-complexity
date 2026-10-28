"""Define how the words of rewriting systems are read, as modules of symbols with parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ._module_spec import Letters, ModuleSpec


def parse_word(value: Letters, names: Sequence[str] = ()) -> list[ModuleSpec]:
    """Return the modules of a word given as a string of one-character symbols, a module, or a sequence of both.

    A string item of a sequence is one symbol when it is a declared name or one character long, and a string of
    one-character symbols otherwise.

    Args:
        value:
            The word.

        names:
            The declared symbol names, which may be longer than one character.

    Returns:
        modules:
            The modules of the word.

    Examples:
        >>> from skcomplex.models.mathematics.fractals._words import parse_word
        >>> from skcomplex.models.mathematics.fractals import ModuleSpec
        >>> [module.name for module in parse_word(['F', ModuleSpec('A', (1.0,)), '+-'])]
        ['F', 'A', '+', '-']
    """
    if isinstance(value, ModuleSpec):
        return [value]
    if isinstance(value, str):
        return [ModuleSpec(value)] if value in names else [ModuleSpec(item) for item in value]
    return [module for item in value for module in parse_word(item, names)]
