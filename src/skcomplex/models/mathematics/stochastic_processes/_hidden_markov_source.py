"""Source of symbols emitted from the states of a hidden Markov chain, one symbol per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class HiddenMarkovSourceElement(Element):
    """Source of symbols emitted from the states of a hidden Markov chain, one symbol per step.

    Args:
        label:
            The label of the source.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        hidden:
            The initial hidden state.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import HiddenMarkovSourceElement
        >>> HiddenMarkovSourceElement('source', hidden=1).hidden
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        hidden: Annotated[float, StateSpec(domain=NN)] = 0,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hidden = hidden
        self.symbol = symbol
