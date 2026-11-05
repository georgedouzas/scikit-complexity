"""Agent of the voter system variants: an imitator, a threshold voter, or a contrarian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._variant_voter_view import VariantVoterView


class VariantVoterElement(Element):
    """Agent of the voter system variants: an imitator, a threshold voter, or a contrarian.

    With a threshold of 0 the voting copies a random neighbour into the agent, as in the voter system. With a positive
    threshold it gives the agent the opinion opposite to its own when the share of neighbours that hold it exceeds the
    threshold, so a threshold of 0.5 gives the majority rule. A contrarian adopts the opposite of what it would
    otherwise adopt.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, 0 or 1.

        contrarian:
            Whether the agent is a contrarian, 1, or not, 0.

        threshold:
            The share of disagreeing neighbours above which the agent changes its opinion, or 0 to copy a random
            neighbour.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import VariantVoterElement
        >>> group = GroupElement('voters', element=VariantVoterElement, size=10, values={'threshold': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('voters', 'VariantVoterElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=NN)] = 0.0,
        contrarian: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.contrarian = contrarian
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[VariantVoterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return VariantVoterView
