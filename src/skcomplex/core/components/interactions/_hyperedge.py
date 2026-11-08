"""Define hyperedge interactions, whose laws run for every hyperedge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ...errors import ValidationError
from ..base import BasePart
from ._base import BaseInteraction

MIN_SIZE = 2


class HyperedgeInteraction(BaseInteraction):
    """Base class of interactions whose laws run once for every hyperedge.

    The hyperedges are those of the network of the nodes a role binds. The first role binds a group with one member per
    node of a network space with hyperedges. A second role may bind a group on the hyperedges of the same network, or a
    group on the edges of one of its layers. The laws are written for one hyperedge of `size` members and read:

    - `namespace.member`, the focal member, whose states rate, noise, and jump laws change,
    - `namespace.others`, the other members in the order of the hyperedge, a tuple of namespaces,
    - `namespace.members`, the members of the hyperedge, which `total`, `mean`, `count`, `maximum`, and `minimum`
      reduce with `over=namespace.members`,
    - `namespace.hyperedge`, the member of the group on hyperedges,
    - `namespace.edge`, the member of the group on edges that joins the first two members of the hyperedge,
    - the parameters of the interaction.

    A law of the focal member runs once per member of every hyperedge and adds into that member. A law that targets the
    member of the group on hyperedges or on edges runs once per hyperedge, with the first member of the hyperedge as the
    focal member. Without a size, the laws run on the hyperedges of every size, one size at a time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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

    def _check(self) -> None:
        """Raise an error for declared states or a size below two.

        Raises:
            ValidationError:
                The type declares states, or the size is not a whole number of at least two.
        """
        super()._check()
        if self.size is None:
            return
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < MIN_SIZE:
            error_msg = (
                f'The size of the hyperedges of {self.label!r} is {self.size!r}. Give a whole number of 2 or more.'
            )
            raise ValidationError(error_msg)
