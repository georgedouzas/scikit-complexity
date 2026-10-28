"""Define the module spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from ..errors import RewritingError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ModuleSpec(BaseSpec):
    """Spec of a symbol of a word with numeric or symbolic parameters, such as `A(x)` or `F(1.0)`.

    Args:
        name:
            The name of the symbol.

        parameters:
            The parameters, numbers in a word and symbols or expressions in a production.

    Examples:
        >>> from skcomplex.specs import ModuleSpec
        >>> ModuleSpec('F', (0.5,)).check().parameters
        (0.5,)
    """

    name: str
    parameters: tuple[object, ...] = ()

    def _check(self) -> Self:
        """Raise an error for a name that is not a nonempty string.

        Returns:
            spec:
                The spec itself.

        Raises:
            RewritingError:
                The name is not a nonempty string.
        """
        if not isinstance(self.name, str) or not self.name:
            error_msg = f'The name of a module must be a nonempty string, but it is {self.name!r}.'
            raise RewritingError(error_msg)
        return self


type Letters = str | ModuleSpec | Sequence[str | ModuleSpec]
