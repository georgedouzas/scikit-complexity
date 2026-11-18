"""Define the random partition of a group, a binding of interactions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...errors import ValidationError
from ...specs._base_spec import BaseSpec

MIN_SIZE = 2


@dataclass(frozen=True)
class RandomPartition(BaseSpec):
    """A random partition of a group into small groups, drawn anew at every step.

    An interaction bound with `along=RandomPartition(3)` shuffles the members of the group `among` at every step, cuts
    them into groups of three, and runs its laws once for every member of every group. The members left over form one
    smaller group. The laws read `s.member`, the member they change, `s.others`, the other members of its group, and
    `s.members`, every member of the group, which reductions read with `over=s.members`.

    Unlike `along='hyperedges'`, whose groups the network fixes, it draws new groups at every step. Choose it when
    members meet in random small groups rather than along fixed links.

    It appears in many fields. Voters discuss in small groups, players of a public goods game form groups every round,
    and animals meet in random encounters.

    Args:
        size:
            The number of members of a group, at least 2.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction, RandomPartition
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> from skcomplex.core.components.laws.utils import mean, where
        >>> from skcomplex.core.systems import System
        >>> def adopt_majority(s):
        ...     return [UpdateLaw(s.member.opinion, where(mean(s.members.opinion, over=s.members) > 0.5, 1, 0))]
        >>> talks = Interaction('talks', among='citizens', along=RandomPartition(3), laws=adopt_majority)
        >>> citizens = Element('citizens', values={'opinion': [1.0, 1.0, 0.0]}, size=3)
        >>> system = System(elements=[citizens], interactions=[talks]).prepare()
        >>> system.simulate(until=1, random_state=0).get_values('citizens.opinion')[-1].tolist()
        [1.0, 1.0, 1.0]
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
