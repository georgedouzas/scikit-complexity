"""Catalytic link from a species to a species that catalyses it, which carries no state of its own."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import Element


class CatalyticLinkElement(Element):
    """Catalytic link from a species to a species that catalyses it, which carries no state of its own.

    Args:
        label:
            The label of the link or of the group of links.

    Examples:
        >>> from skcomplex.models.biology.evolution import CatalyticLinkElement
        >>> CatalyticLinkElement('link').label
        'link'
    """
