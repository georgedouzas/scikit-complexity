"""Define the interaction requirement spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ._requirement_spec import RequirementSpec
from ._role_spec import RoleSpec


@dataclass(frozen=True)
class InteractionRequirementSpec(RequirementSpec):
    """What an interaction requires.

    It holds what each of its roles needs, and the laws it returns with their time and space. An empty `time` or `space`
    accepts any class, and empty `laws` means that only running the interaction shows them.

    Args:
        time:
            The time classes its laws run in, or empty when one of them runs in any.

        space:
            The space classes its laws run in, or empty when one of them runs in any.

        roles:
            The role of every name, whose `states` hold the names of the states it reads for this interaction.

        laws:
            The law classes the interaction may return.

    Examples:
        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> needs = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: s.target.n).get_requirements()
        >>> needs.roles['target'].states
        ('n',)
        >>> [law.__name__ for law in needs.laws]
        ['RateLaw', 'NoiseLaw', 'FractionalNoiseLaw']
        >>> [time.__name__ for time in needs.time]
        ['ContinuousTime']
    """

    roles: Mapping[str, RoleSpec] = field(default_factory=dict)
    laws: tuple[type, ...] = ()
