"""Cell that takes the majority opinion of its Moore neighbourhood, itself included."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class MajorityVoterElement(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood, itself included.

    The majority rule is a `LocalRuleInteraction` of the lattice, which `majority_rule` adds.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import MajorityVoterElement
        >>> MajorityVoterElement('cell', opinion=1).opinion
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        opinion: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
