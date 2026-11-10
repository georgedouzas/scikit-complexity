"""Define the interaction whose laws run once for every hyperedge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ...errors import ValidationError
from ..base import BasePart
from ..spaces import BaseSpace
from ._base import BaseInteraction

MIN_SIZE = 2


class HyperedgeInteraction(BaseInteraction):
    """An interaction whose laws run once for every hyperedge of a network.

    A hyperedge is a link that joins more than two nodes, such as a group of three friends. The first role binds a group
    with one member per node of a network with hyperedges. A second role may bind a group on the hyperedges, or on the
    edges of one layer. The laws are written for one hyperedge of `size` members and read:

    - `namespace.member`, the focal member, whose states rate, noise, and jump laws change,
    - `namespace.others`, the other members in the order of the hyperedge, a tuple of namespaces,
    - `namespace.members`, all the members, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with
      `over=namespace.members`,
    - `namespace.hyperedge`, the member of the group on hyperedges,
    - `namespace.edge`, the member of the group on edges that joins the first two members,
    - the parameters of the interaction.

    A law of the focal member runs once per member of every hyperedge. A law that targets the member on a hyperedge or
    an edge runs once per hyperedge, with the first member as the focal member.

    Unlike `NeighborInteraction`, which sees the two ends of an edge, it sees every member of a group link at once.
    Choose it when a change depends on a whole group, such as a majority of three, and not on pairs alone.

    It appears in many fields. Group conversations shift opinions, a disease spreads within households and classrooms,
    and three species compete at once in higher-order ecological models.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'consensus'`.

        size:
            The number of members of the hyperedges the laws run on, or `None` for every size.

    Raises:
        ValidationError:
            The size is not a whole number of at least two.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.interactions import HyperedgeInteraction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.laws.utils import mean
        >>> from skcomplex.core.specs import RoleSpec
        >>> class ConsensusInteraction(HyperedgeInteraction):
        ...     def __init__(self, label, voters: Annotated[str, RoleSpec()], size=None):
        ...         super().__init__(label, size=size)
        ...         self.voters = voters
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.member.x, mean(s.members.x, over=s.members) - s.member.x)]
        >>> ConsensusInteraction('consensus', voters='voters', size=3).size
        3
    """

    def __init__(self, label: str, size: int | None = None) -> None:
        super().__init__(label)
        self.size = size

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the interaction for one hyperedge, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for declared states or a size below two.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, or the size is not a whole number of at least two.
        """
        super()._prepare(context)
        if self.size is None:
            return
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < MIN_SIZE:
            error_msg = (
                f'The size of the hyperedges of {self.label!r} is {self.size!r}. Give a whole number of 2 or more.'
            )
            raise ValidationError(error_msg)
