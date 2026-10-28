"""Interaction of a compiled problem with the paths its roles are bound to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ...core.interactions import BaseInteraction
from ...specs import BaseSpec


@dataclass(frozen=True)
class CompiledInteraction(BaseSpec):
    """Interaction of a compiled problem with the paths its roles are bound to.

    Args:
        path:
            The slash-separated path of the interaction in the system tree.

        interaction:
            The interaction.

        roles:
            A mapping from role name to the path of the element, group, or observable bound to it, or to the paths
            of the elements of an element set.
    """

    path: str
    interaction: BaseInteraction
    roles: Mapping[str, str | tuple[str, ...]]
