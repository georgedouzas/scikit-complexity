"""Define the law of one production of a string rewriting system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from ....core.components.laws import BaseLaw, RequirementSpec, UpdateLaw
from ....core.errors import RewritingError
from ._module_spec import Letters, ModuleSpec


@dataclass(frozen=True)
class ProductionLaw(BaseLaw):
    """A law that rewrites every matching symbol of a word by a successor.

    A string rewriting system, or L-system, grows a word by replacing each symbol by a string at every generation. A
    interaction that binds a word sets its productions, and every generation applies all of them at once. The
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

    Examples:
        >>> from skcomplex.systems.general.rewriting import ProductionLaw
        >>> ProductionLaw('F', 'F+F--F+F').successor
        'F+F--F+F'
    """

    law_tag: ClassVar[str] = 'update'

    predecessor: str | ModuleSpec
    successor: Letters
    _: KW_ONLY
    left: Letters | None = None
    right: Letters | None = None
    condition: object = None
    probability: object = None
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states.

        A production rewrites the word of its element as a whole.
        """
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: the discrete time of an update."""
        return UpdateLaw.get_requirements()

    def _check(self) -> None:
        """Raise an error for a predecessor not one module, or a bad numeric probability.

        A numeric probability must be in `[0, 1]`.
        """
        super()._check()
        if len(ModuleSpec.parse_word(self.predecessor)) != 1:
            error_msg = f'The predecessor of a production is one symbol or module, not {self.predecessor!r}.'
            raise RewritingError(error_msg)
        for module in ModuleSpec.parse_word(self.successor):
            module._check()
        if isinstance(self.probability, numbers.Real) and not 0 <= float(self.probability) <= 1:
            error_msg = f'The probability of a production is in [0, 1], not {self.probability!r}.'
            raise RewritingError(error_msg)

    def _get_value(self, inputs: object) -> object:
        """Return the modules of the successor.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value, unused.

        Returns:
            modules:
                The modules of the successor.
        """
        return ModuleSpec.parse_word(self.successor)
