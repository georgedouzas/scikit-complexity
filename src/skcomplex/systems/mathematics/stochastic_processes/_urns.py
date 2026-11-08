"""Stochastic process systems of urns with reinforcement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import JumpInteraction
from ....core.components.system import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec


class PolyaUrnElement(Element):
    """Urn of red and blue balls.

    A drawn ball returns with `added` extra balls of its color.

    The jumps of the system draw at rate 1, and a draw picks a color with probability proportional to its number of
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
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnElement
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


class PolyaUrnSystem(BaseSystem):
    """Independent Pólya urns.

    The share of red balls converges to a random limit with a beta distribution.

    The limit follows `Beta(red / added, blue / added)`, uniform for one ball of each color and one added ball. The
    system has the states `urns.red` and `urns.blue`.

    Args:
        n_urns:
            The number of independent urns.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PolyaUrnSystem
        >>> PolyaUrnSystem(3).prepare().system_.elements[0].prepare().size_
        3
    """

    field = 'mathematics/stochastic_processes'
    topics = ('urn', 'reinforcement', 'jump process', 'history dependence')
    references = ('Eggenberger and Pólya 1923', 'Thurner et al. 2018')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        n_urns: int = 10,
        *,
        red: int = 1,
        blue: int = 1,
        added: int = 1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_urns = n_urns
        self.red = red
        self.blue = blue
        self.added = added

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        urns = GroupElement(
            'urns',
            element=PolyaUrnElement,
            size=self.n_urns,
            values={'red': float(self.red), 'blue': float(self.blue), 'added': self.added},
        )
        red_draws = JumpInteraction(
            'red_draws',
            target='urns',
            changes={'red': self.added},
            rate=lambda s: s.target.red / (s.target.red + s.target.blue),
        )
        blue_draws = JumpInteraction(
            'blue_draws',
            target='urns',
            changes={'blue': self.added},
            rate=lambda s: s.target.blue / (s.target.red + s.target.blue),
        )
        return System(self.label, elements=[urns], interactions=[red_draws, blue_draws])
