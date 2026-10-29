"""Define production laws, the productions of string rewriting systems on words."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers

from ....core.laws import BaseLaw
from ....core.spaces import DiscreteTime
from ....specs import RequirementSpec
from ._module_spec import Letters, ModuleSpec
from ._rewriting_error import RewritingError
from ._words import parse_word


class ProductionLaw(BaseLaw):
    """Production of a string rewriting system, which replaces every matching module of a word by a successor.

    A word element returns its productions from `_get_laws`, and every generation applies all of them at once. The
    predecessor's parameters are symbols bound to the parameters of each matching module. The condition, the
    successor's parameters, and the probability are expressions of these symbols and of the parameters of the element.
    A production with a left or a right context matches only modules whose neighbours, skipping the ignored symbols of
    the word and its bracketed branches, spell the context.

    Args:
        predecessor:
            The symbol, or a module whose parameters are symbols.

        successor:
            The word that replaces the predecessor.

        left:
            The left context, or `None`.

        right:
            The right context, or `None`.

        condition:
            A symbolic condition of the predecessor's parameters, or `None`.

        probability:
            The probability of a stochastic production, or `None` for a deterministic one. The probabilities of the
            stochastic productions of one predecessor, context, and condition sum to one.

        label:
            The name of the law, or the empty string for a law without one.

    Attributes:
        value_:
            The modules of the successor, set by `run`.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import ProductionLaw
        >>> [module.name for module in ProductionLaw('F', 'F+F--F+F').check().run({}).value_]
        ['F', '+', 'F', '-', '-', 'F', '+', 'F']
    """

    value_: list[ModuleSpec]

    def __init__(
        self,
        predecessor: str | ModuleSpec,
        successor: Letters,
        *,
        left: Letters | None = None,
        right: Letters | None = None,
        condition: object = None,
        probability: object = None,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.predecessor = predecessor
        self.successor = successor
        self.left = left
        self.right = right
        self.condition = condition
        self.probability = probability

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a production rewrites the word of its element as a whole."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> None:
        """Raise an error for a predecessor that is not one module, or a numeric probability outside `[0, 1]`."""
        if len(parse_word(self.predecessor)) != 1:
            error_msg = f'The predecessor of a production is one symbol or module, not {self.predecessor!r}.'
            raise RewritingError(error_msg)
        for module in parse_word(self.successor):
            module.check()
        if isinstance(self.probability, numbers.Real) and not 0 <= float(self.probability) <= 1:
            error_msg = f'The probability of a production is in [0, 1], not {self.probability!r}.'
            raise RewritingError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_`, the modules of the successor.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, unused.
        """
        self.value_ = parse_word(self.successor)
