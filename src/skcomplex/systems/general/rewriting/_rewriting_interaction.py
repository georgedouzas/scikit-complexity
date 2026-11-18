"""Define the interaction that rewrites words and patches by given rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import ClassVar

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw
from ....core.components.spaces import BaseSpace
from ....core.errors import RewritingError
from ....core.specs import Role
from ._production import ProductionLaw
from ._substitution import SubstitutionLaw


class RewritingInteraction(Interaction):
    """An interaction that rewrites a word or a patch by the rules it is given.

    At every step each symbol of a word is replaced by the successor of its `ProductionLaw`, and each tile of a patch by
    the pieces of its `SubstitutionLaw`, all at once.

    Unlike a subclass of `Interaction` that sets its rules in `_prepare`, it takes fixed rules. Choose it when the rules
    read no parameters of their own, as the algae of Lindenmayer, and write a subclass for rules that do, such as the
    ratio of a Cantor set.

    It appears in many fields. The productions of an L-system grow a plant in biology, a substitution builds a fractal
    in mathematics, and the Thue-Morse word of computer science doubles at every step.

    Args:
        label:
            The name of the interaction, such as `'growth'`.

        target:
            The path of the word or patch it rewrites.

        laws:
            The productions of a word or the substitutions of a patch.

    Raises:
        RewritingError:
            A law is neither a production nor a substitution.

    Examples:
        In Lindenmayer's algae `A` becomes `AB` and `B` becomes `A`, so the length of the word runs through the
        Fibonacci numbers.

        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.rewriting import ProductionLaw, RewritingInteraction, WordElement
        >>> laws = [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> growth = RewritingInteraction('growth', target='algae', laws=laws)
        >>> system = System(elements=[WordElement('algae', axiom='A')], interactions=[growth])
        >>> system.prepare().simulate(until=5).get_values('algae.size').tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ProductionLaw, SubstitutionLaw)

    def __init__(self, label: str, *, target: Role, laws: Sequence[ProductionLaw | SubstitutionLaw] = ()) -> None:
        super().__init__(label)
        self.target = target
        self.laws = laws

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Raise an error for a law that rewrites nothing, and set the laws of a binding.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the word or patch it binds.

        Raises:
            RewritingError:
                A law is neither a production nor a substitution.
        """
        super()._prepare(context)
        wrong = [law for law in self.laws if not isinstance(law, ProductionLaw | SubstitutionLaw)]
        if wrong:
            error_msg = f'{self.label!r} rewrites by {wrong[0]!r}. Give ProductionLaw or SubstitutionLaw laws.'
            raise RewritingError(error_msg)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = list(self.laws)
