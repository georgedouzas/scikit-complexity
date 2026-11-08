"""Define the module spec, a symbol of a word with its parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..errors import RewritingError
from ._base_spec import BaseSpec

type Letters = str | ModuleSpec | Sequence[str | ModuleSpec]


@dataclass(frozen=True)
class ModuleSpec(BaseSpec):
    """Spec of a symbol of a word with numeric or symbolic parameters.

    Examples are `A(x)` or `F(1.0)`.

    Args:
        name:
            The name of the symbol.

        parameters:
            The parameters, numbers in a word and symbols or expressions in a production.

    Examples:
        >>> from skcomplex.core.specs import ModuleSpec
        >>> ModuleSpec('F', (0.5,)).parameters
        (0.5,)
    """

    name: str
    parameters: tuple[float | Expression, ...] = ()

    def _check(self) -> None:
        """Raise an error for a name that is not a nonempty string.

        Raises:
            RewritingError:
                The name is not a nonempty string.
        """
        if not isinstance(self.name, str) or not self.name:
            error_msg = f'The name of a module must be a nonempty string, but it is {self.name!r}.'
            raise RewritingError(error_msg)

    @classmethod
    def parse_word(cls, value: Letters, names: Sequence[str] = ()) -> list[ModuleSpec]:
        """Return the modules of a word.

        The word is a string of one-character symbols, a module, or a sequence of both. A string item of a sequence is
        one symbol when it is a declared name or one character long, and a string of one-character symbols otherwise.

        Args:
            value:
                The word.

            names:
                The declared symbol names, which may be longer than one character.

        Returns:
            modules:
                The modules of the word.

        Examples:
            >>> from skcomplex.core.specs import ModuleSpec
            >>> [module.name for module in ModuleSpec.parse_word(['F', ModuleSpec('A', (1.0,)), '+-'])]
            ['F', 'A', '+', '-']
        """
        if isinstance(value, cls):
            return [value]
        if isinstance(value, str):
            return [cls(value)] if value in names else [cls(item) for item in value]
        return [module for item in value for module in cls.parse_word(item, names)]
