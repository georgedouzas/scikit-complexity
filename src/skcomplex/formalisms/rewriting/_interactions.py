"""Define rewriting interactions, which return the productions applied to their targets in parallel."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

from ...core.base import (
    role,
)
from ...core.interactions import BaseInteraction
from ...core.laws import Symbols
from ._laws import RewritingLaw


class Rewriting(BaseInteraction):
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
        >>> from skcomplex.formalisms.rewriting import Rewriting
        >>> from skcomplex.formalisms.rewriting import Module, ProductionLaw
        >>> from skcomplex.core.base import symbol
        >>> class HalvingInteraction(Rewriting):
        ...     ratio: float = parameter(0.5)
        ...
        ...     def productions(self, s):
        ...         x = symbol('x')
        ...         return [ProductionLaw(Module('A', x), [Module('F', x), Module('A', s.ratio * x)])]
        >>> HalvingInteraction('halving', target='plant').ratio
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

        rules:
            The productions, substitutions, or graph productions.

        **values:
            The values of the declared variables, by name:

            target:
                The path of the rewritten element.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Rewrite
        >>> from skcomplex.formalisms.rewriting import ProductionLaw
        >>> Rewrite('rules', target='algae', rules=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]).rules[1]
        B → A
    """

    def __init__(
        self,
        label: str,
        *,
        rules: Sequence[RewritingLaw] = (),
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.rules = rules

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
