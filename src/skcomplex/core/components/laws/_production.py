"""Define the law of one production of a string rewriting system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from typing import ClassVar

from ...errors import RewritingError
from ...specs import Letters, ModuleSpec, RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class ProductionLaw(BaseLaw):
    """A law that rewrites every matching symbol of a word by a successor.

    A string rewriting system, or L-system, grows a word by replacing each symbol by a string at every generation. A
    word element returns its productions from `_get_laws`, and every generation applies all of them at once. The
    parameters of the predecessor are symbols bound to the parameters of each matching symbol. The condition, the
    parameters of the successor, and the probability are expressions of these symbols and of the parameters of the
    element. A production with a left or a right context matches only where the neighbours spell it, skipping ignored
    symbols and bracketed branches.

    Unlike `UpdateLaw`, which changes the value of a fixed number of states, it replaces symbols by strings, so the word
    grows. Choose it when a structure develops by repeated rewriting.

    It appears in many fields. Lindenmayer systems model the branching of plants, rewriting rules build fractal curves,
    and grammars generate the structure of sentences.

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
        >>> from skcomplex.core.components.laws import ProductionLaw
        >>> [module.name for module in ProductionLaw('F', 'F+F--F+F').prepare().run({}).value_]
        ['F', '+', 'F', '-', '-', 'F', '+', 'F']
    """

    law_tag: ClassVar[str] = 'update'

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
        """Return no states.

        A production rewrites the word of its element as a whole.
        """
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a predecessor not one module, or a bad numeric probability.

        A numeric probability must be in `[0, 1]`.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if len(ModuleSpec.parse_word(self.predecessor)) != 1:
            error_msg = f'The predecessor of a production is one symbol or module, not {self.predecessor!r}.'
            raise RewritingError(error_msg)
        for module in ModuleSpec.parse_word(self.successor):
            module._check()
        if isinstance(self.probability, numbers.Real) and not 0 <= float(self.probability) <= 1:
            error_msg = f'The probability of a production is in [0, 1], not {self.probability!r}.'
            raise RewritingError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_`, the modules of the successor.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, unused.
        """
        self.value_ = ModuleSpec.parse_word(self.successor)
