"""Define the interaction requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...specs._role_spec import RoleSpec
from ..laws import RequirementSpec


@dataclass(frozen=True)
class _InteractionRequirementSpec(RequirementSpec):
    """A spec of what an interaction requires of its roles, laws, time, and space.

    It holds what each of its roles needs, and the laws it returns with their time and space. An empty `time` or `space`
    accepts any class, and empty `laws` means that only running the interaction shows them.

    Args:
        time:
            The time classes its laws run in, or empty when one of them runs in any.

        space:
            The space classes its laws run in, or empty when one of them runs in any.

        sites:
            Whether one of its laws also runs on any other space with sites.

        roles:
            The role of every name, whose `states` hold the names of the states it reads for this interaction.

        laws:
            The law classes the interaction may return.

    Examples:
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> growth = Interaction('growth', roles={'target': 'rabbits'}, laws=lambda s: [RateLaw(s.target.n, 1.0)])
        >>> list(growth.get_requirements().roles)
        ['target']
    """

    roles: Mapping[str, RoleSpec] = field(default_factory=dict)
    laws: tuple[type, ...] = ()
