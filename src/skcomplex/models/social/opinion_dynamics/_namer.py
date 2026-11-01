"""Agent of the naming game, with an inventory of the words it knows for one object."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._namer_view import NamerView


class NamerElement(Element):
    """Agent of the naming game, with an inventory of the words it knows for one object.

    The inventory is an indicator vector over the lexicon, with one slot per agent for the word that agent invents
    when it first speaks with an empty inventory.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        words:
            The initial inventory, 1 for every known word and 0 otherwise.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> import numpy as np
        >>> from skcomplex.models.social.opinion_dynamics import NamerElement
        >>> group = GroupElement('namers', element=NamerElement, size=5, values={'words': np.zeros((5, 5))})
        >>> group.label, group.element.__name__, group.size
        ('namers', 'NamerElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        words: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.words = words

    @classmethod
    def _get_view(cls) -> type[NamerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return NamerView
