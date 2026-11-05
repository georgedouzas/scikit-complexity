"""Discussions of the Galam model, which split the citizens into random groups that adopt their majority opinion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import GroupingInteraction
from ....core.laws.utils import mean, where
from ....specs import RoleSpec


class DiscussionsInteraction(GroupingInteraction):
    """Discussions of the Galam model, which split the citizens into random groups that adopt their majority opinion.

    It is a symbolic `GroupingInteraction` of random partitions, drawn again at every step. A tie in a group of even
    size goes to opinion 0, the status quo.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of citizens.

        size:
            The number of citizens in each discussion group, the last group holding the remainder.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import DiscussionsInteraction
        >>> DiscussionsInteraction('talks', target='citizens', size=4).write
        'opinion'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], size: int = 3) -> None:
        super().__init__(
            label,
            target=target,
            groups='partition',
            size=size,
            form='symbolic',
            write='opinion',
            function=self._get_majority,
        )

    def _get_majority(self, group: SimpleNamespace) -> Expression:
        """Return 1 when more than half of the members of a group hold opinion 1, and 0 otherwise."""
        return where(mean(group.members.opinion, over=group.members) > 1 / 2, 1, 0)
