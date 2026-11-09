"""Urn systems.

The module holds the Ehrenfest system of balls moved one at a time between two boxes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class EhrenfestUrnElement(Element):
    """First box of the Ehrenfest urn, holding `n` of the `N` balls.

    One of the balls moves to the other box every step.

    The count falls by one with the probability `n / N` that the drawn ball is in the box, and rises by one otherwise.

    Args:
        label:
            The label of the box.

        visual:
            How the box is drawn, or `None` for `default_visual_`.

        n:
            The initial number of balls in the box.

        N:
            The number of balls in both boxes.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EhrenfestUrnElement
        >>> EhrenfestUrnElement('urn', n=10).N
        40
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[int, StateSpec(domain=NN)] = 0,
        N: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 40,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.N = N

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the move of one ball.

        The ball moves out of the box with probability `n / N` and into it otherwise.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of the count.
        """
        s = namespace
        return [UpdateLaw(s.n, s.n + 1 - 2 * bernoulli('pick', s.n / s.N))]


class EhrenfestUrnSystem(BaseSystem):
    """The Ehrenfest urn system.

    Balls move one at a time between two boxes, and the system relaxes to equal counts.

    At every step one of the `N` numbered balls is drawn at random and moved to the other box, so the count `n` of the
    first box falls by one with probability `n / N` and rises by one otherwise. Started with all balls in the first box,
    the count relaxes to `N / 2` with the exponential mean `N / 2 + (n_0 - N / 2) (1 - 2 / N)^t`, and then fluctuates
    around it: the chain is reversible, yet the relaxation looks irreversible. Its stationary distribution is the
    binomial distribution of `N` balls with probability one half. The system has the count `urn.n` of balls in the first
    box.

    Args:
        n_balls:
            The number of balls `N`.

        in_first:
            The initial number of balls in the first box, or `None` for all of them.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EhrenfestUrnSystem
        >>> EhrenfestUrnSystem(10).prepare().system_.elements[0].n
        10
    """

    field = 'physics/statistical_mechanics'
    topics = ('markov chain', 'irreversibility', 'entropy', 'discrete map')
    references = ('Ehrenfest and Ehrenfest 1907', 'Barsky 2020', 'Kac 1947')
    until = 200
    dt = 1

    def __init__(
        self,
        n_balls: int = 40,
        *,
        in_first: int | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_balls = n_balls
        self.in_first = in_first

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        n = self.n_balls if self.in_first is None else self.in_first
        return System(self.label, elements=[EhrenfestUrnElement('urn', n=n, N=self.n_balls)])
