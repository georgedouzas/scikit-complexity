"""Edge of a growing network, which holds no state of its own."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.elements import Element


class LinkElement(Element):
    """Edge of a growing network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import LinkElement
        >>> LinkElement('links').label
        'links'
    """
