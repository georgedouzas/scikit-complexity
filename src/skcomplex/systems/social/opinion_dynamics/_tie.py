"""Edge of a coevolving network, which holds no state of its own."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.elements import Element


class TieElement(Element):
    """Edge of a coevolving network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import TieElement
        >>> TieElement('ties').label
        'ties'
    """
