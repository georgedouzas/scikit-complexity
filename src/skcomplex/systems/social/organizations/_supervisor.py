"""Supervisor of the insider sabotage system, who wanders between adjacent cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.elements import Element


class SupervisorElement(Element):
    """Supervisor of the insider sabotage system, who wanders between adjacent cells.

    The wandering is a `MovementInteraction` to a random adjacent cell, which `insider_sabotage` adds.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import SupervisorElement
        >>> group = GroupElement('supervisors', element=SupervisorElement, size=3)
        >>> group.label, group.element.__name__, group.size
        ('supervisors', 'SupervisorElement', 3)
    """
