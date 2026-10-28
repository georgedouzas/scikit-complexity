"""Define hyperedge interactions, whose laws run for every hyperedge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction


class HyperedgeInteraction(BaseInteraction):
    """Base class of interactions whose laws run once for every hyperedge of the network of the nodes a role binds.

    The laws are written for one hyperedge and read the members on it and the parameters.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions import HyperedgeInteraction
        >>> from skcomplex.specs import RoleSpec
        >>> class ConsensusInteraction(HyperedgeInteraction):
        ...     def __init__(self, label, voters: Annotated[str, RoleSpec()]):
        ...         super().__init__(label)
        ...         self.voters = voters
        >>> ConsensusInteraction('consensus', voters='voters').voters
        'voters'
    """

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the laws of the interaction for one hyperedge, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
