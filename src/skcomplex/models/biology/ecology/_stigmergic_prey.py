"""Prey that eats plants and fears the predators around it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ._forager import ForagerElement


class StigmergicPreyElement(ForagerElement):
    """Prey that eats plants and fears the predators around it.

    Args:
        label:
            The label of the agent or group.

        age:
            The steps since the animal last bred.

        hunger:
            The steps since the animal last ate.

        happiness:
            The happiness, which eating raises.

        distress:
            The fear, which predators around it raise.

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
        >>> from skcomplex.models.biology.ecology import StigmergicPreyElement
        >>> StigmergicPreyElement.food
        'plants'
    """
