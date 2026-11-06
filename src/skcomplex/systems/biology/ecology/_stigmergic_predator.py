"""Predator that eats prey and grows angry with hunger."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.specs import VisualSpec
from ._forager import ForagerElement

PREY = 'prey'


class StigmergicPredatorElement(ForagerElement):
    """Predator that eats prey and grows angry with hunger.

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
            The anger, which hunger raises.

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
        >>> from skcomplex.systems.biology.ecology import StigmergicPredatorElement
        >>> StigmergicPredatorElement.food
        'prey'
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'distress', 'radius': 0.45}),
        *,
        age: float = 0.0,
        hunger: float = 0.0,
        happiness: float = 0.0,
        distress: float = 0.0,
        experience: float = 0.0,
        maturity: float = 5.0,
        reproduction: float = 0.3,
        starvation: float = 10.0,
        memory: float = 0.3,
        communication: float = 0.1,
        emotions: float = 1.0,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            age=age,
            hunger=hunger,
            happiness=happiness,
            distress=distress,
            experience=experience,
            maturity=maturity,
            reproduction=reproduction,
            starvation=starvation,
            memory=memory,
            communication=communication,
            emotions=emotions,
        )

    food: ClassVar[str] = PREY
    happiness_mark: ClassVar[str] = 'predator_happiness'
    distress_mark: ClassVar[str] = 'predator_anger'
