"""Animal that hunts its food guided by its emotions and those its kind left on the ground, and breeds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._forager_view import ForagerView

PLANTS = 'plants'


class ForagerElement(Element):
    """Animal that hunts its food guided by its emotions and those its kind left on the ground, and breeds.

    Every step the animal ages and grows hungry, and starves when its hunger passes `starvation`. It moves to the
    neighbouring cell, empty or holding food, of highest preference: 1 for food, plus, with emotions, the happiness
    marked there minus its own and its own distress minus the distress marked there. It eats food it moves onto.
    Its emotions follow `E' = (1 - memory) (X + communication C) + memory E`, with the experience `X` of the step and
    the mean emotion `C` marked around it, and it marks its cell with them. A mature animal breeds into an empty
    neighbouring cell with a probability that happiness raises and distress lowers, and its offspring shares its
    hunger, so a lineage starves together when its food is gone. Subclasses name their food and the marks they read,
    and the model adds the ageing, the hunger, the starvation, the experience of distress, the move, the meal, the
    emotions, the marks, and the breeding as member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear or anger.

        experience:
            The distress the animal experienced in the step, before it moved.

        maturity:
            The age at which the animal can breed.

        reproduction:
            The probability of breeding at maturity without emotions.

        starvation:
            The hunger at which the animal dies.

        memory:
            The share of its previous emotion that an animal keeps.

        communication:
            The weight of the emotions marked around the animal.

        emotions:
            Whether emotions guide the animal, 1 or 0.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import StigmergicPreyElement
        >>> GroupElement('prey', element=StigmergicPreyElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'happiness', 'radius': 0.35}),
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        hunger: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        happiness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        distress: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        experience: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        starvation: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        memory: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        communication: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        emotions: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.hunger = hunger
        self.happiness = happiness
        self.distress = distress
        self.experience = experience
        self.maturity = maturity
        self.reproduction = reproduction
        self.starvation = starvation
        self.memory = memory
        self.communication = communication
        self.emotions = emotions

    food: ClassVar[str] = PLANTS
    happiness_mark: ClassVar[str] = 'prey_happiness'
    distress_mark: ClassVar[str] = 'prey_fear'

    @classmethod
    def _get_view(cls) -> type[ForagerView]:
        """Return the view of a forager or a predator, `ForagerView`."""
        return ForagerView
