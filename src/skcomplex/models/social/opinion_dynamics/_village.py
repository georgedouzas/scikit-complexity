"""Village of Axelrod's model, with a culture of `F` features of `q` traits each, stored as one integer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._village_view import VillageView


class VillageElement(Element):
    """Village of Axelrod's model, with a culture of `F` features of `q` traits each, stored as one integer.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        culture:
            The culture code, the traits of the features as the digits of a number in base `q`.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import VillageElement
        >>> GroupElement(
        ...     'villages', element=VillageElement, size=4, values={'culture': 0, 'features': 3, 'traits': 5}
        ... ).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'culture'}),
        *,
        culture: Annotated[float, StateSpec(domain=NN)] = 0,
        features: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        traits: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.culture = culture
        self.features = features
        self.traits = traits

    @classmethod
    def _get_view(cls) -> type[VillageView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return VillageView
