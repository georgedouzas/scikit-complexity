"""Define the random partition of a group, a binding of interactions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError

MIN_SIZE = 2


@dataclass(frozen=True)
class RandomPartition(BaseSpec):
    """A random partition of a group into small groups, drawn anew at every step.

    An interaction bound with `along=RandomPartition(3)` shuffles the members of the group `among` at every step, cuts
    them into groups of three, and runs its laws once for every member of every group. The members left over form one
    smaller group. The laws read `s.member`, the member they change, `s.others`, the other members of its group, and
    `s.members`, every member of the group. Unlike `along='hyperedges'`, whose groups a network fixes, it draws new
    groups at every step. Voters discuss in small groups, players of a public goods game meet every round, and animals
    meet in random encounters.

    Read more in the user guide, under Interactions.

    Args:
        size:
            The number of members of a group, at least 2.

    See Also:
        Interaction: Runs its laws once for every member of every group.

    Examples:
        >>> from skcomplex.core.components.interactions import Interaction, RandomPartition
        >>> talks = Interaction('talks', among='citizens', along=RandomPartition(3))
        >>> talks.along.size
        3
    """

    size: int

    def _check(self) -> None:
        """Raise an error for a size that is not a whole number of at least 2.

        Raises:
            ValidationError:
                The size is not a whole number of at least 2.
        """
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < MIN_SIZE:
            error_msg = (
                f'The groups of a random partition hold at least 2 members, such as RandomPartition(3), not '
                f'{self.size!r}.'
            )
            raise ValidationError(error_msg)
