"""Define rewriting interactions, which return the productions applied to their targets in parallel."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ..base import (
    role,
)
from ..laws._rewriting import RewritingLaw
from ..laws._symbols import Symbols
from ._interactions import Interaction


class Rewriting(Interaction):
    """Base class of interactions that rewrite the word, patch, or host graph bound to their `target` role.

    Subclasses declare parameters with `parameter` and override `productions`, where `s.<parameter>` is a symbol that
    conditions, successor parameters, probabilities, and affine maps can use.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the rewritten element.

    Examples:
        >>> from skcomplex.core.base import parameter
        >>> from skcomplex.core.interactions import Rewriting
        >>> from skcomplex.core.laws import Module, Production
        >>> from skcomplex.core.base import symbol
        >>> class Halving(Rewriting):
        ...     ratio: float = parameter(0.5)
        ...
        ...     def productions(self, s):
        ...         x = symbol('x')
        ...         return [Production(Module('A', x), [Module('F', x), Module('A', s.ratio * x)])]
        >>> Halving('halving', target='plant').ratio
        0.5
    """

    target: str = role()

    def productions(self, s: Symbols) -> list[RewritingLaw]:
        """Return the productions or substitutions of the interaction.

        Args:
            s:
                The namespace of the parameter symbols.

        Returns:
            productions:
                The productions, substitutions, or graph productions.
        """
        del s
        return []


class Rewrite(Rewriting):
    """Interaction that rewrites its target by fixed productions, substitutions, or graph productions.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the rewritten element.

        rules:
            The productions, substitutions, or graph productions.

    Examples:
        >>> from skcomplex.core.interactions import Rewrite
        >>> from skcomplex.core.laws import Production
        >>> Rewrite('rules', target='algae', rules=[Production('A', 'AB'), Production('B', 'A')]).rules[1]
        B → A
    """

    rules: Sequence[RewritingLaw] = ()

    def productions(self, s: Symbols) -> list[RewritingLaw]:
        """Return the fixed rules.

        Args:
            s:
                The namespace of the parameter symbols.

        Returns:
            productions:
                The rules.
        """
        del s
        return list(self.rules)
