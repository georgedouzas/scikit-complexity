"""Stochastic process models: urns with reinforcement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element, GroupElement
from ....core.interactions import JumpInteraction
from ....core.system import System
from ....specs import StateSpec, VisualSpec
from ..._catalog import register_model


class PolyaUrnElement(Element):
    """Urn of red and blue balls, from which a drawn ball returns with `added` extra balls of its color.

    The jumps of the model draw at rate 1, and a draw picks a color with probability proportional to its number of
    balls, so the sequence of draws is that of Pólya's urn in discrete time.

    Args:
        label:
            The label of the urn or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import PolyaUrnElement
        >>> PolyaUrnElement('urn', added=2).added
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        red: Annotated[float, StateSpec(domain=NN)] = 1.0,
        blue: Annotated[float, StateSpec(domain=NN)] = 1.0,
        added: int = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.red = red
        self.blue = blue
        self.added = added


@register_model(
    field='mathematics/stochastic_processes',
    tags=('urn', 'reinforcement', 'jump process', 'history dependence'),
    references=('Eggenberger and Pólya 1923', 'Thurner et al. 2018'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def polya_urn(n_urns: int = 10, *, red: int = 1, blue: int = 1, added: int = 1) -> System:
    """Return independent Pólya urns, whose share of red balls converges to a random limit with a beta distribution.

    The limit follows `Beta(red / added, blue / added)`, uniform for one ball of each color and one added ball.

    Args:
        n_urns:
            The number of independent urns.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

    Returns:
        system:
            The model, with the states `urns.red` and `urns.blue`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import polya_urn
        >>> polya_urn(3).elements[0].prepare().size_
        3
    """
    urns = GroupElement(
        'urns',
        element=PolyaUrnElement,
        size=n_urns,
        values={'red': float(red), 'blue': float(blue), 'added': added},
    )
    red_draws = JumpInteraction(
        'red_draws',
        target='urns',
        changes={'red': added},
        rate=lambda s: s.target.red / (s.target.red + s.target.blue),
    )
    blue_draws = JumpInteraction(
        'blue_draws',
        target='urns',
        changes={'blue': added},
        rate=lambda s: s.target.blue / (s.target.red + s.target.blue),
    )
    return System(elements=[urns], interactions=[red_draws, blue_draws])
