"""Urn systems: the Ehrenfest system of balls moved one at a time between two boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class EhrenfestUrnElement(Element):
    """First box of the Ehrenfest urn, which holds `n` of the `N` balls, one of which moves to the other box every step.

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
        """Return the move of one ball, out of the box with probability `n / N` and into it otherwise.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of the count.
        """
        s = namespace
        return [UpdateLaw(s.n, s.n + 1 - 2 * bernoulli('pick', s.n / s.N))]


@register_system(
    field='physics/statistical_mechanics',
    tags=('markov chain', 'irreversibility', 'entropy', 'discrete map'),
    references=('Ehrenfest and Ehrenfest 1907', 'Barsky 2020', 'Kac 1947'),
    defaults={'until': 200, 'dt': 1},
)
def ehrenfest_urn(n_balls: int = 40, *, in_first: int | None = None) -> System:
    """Return the Ehrenfest urn system, balls moved one at a time between two boxes, which relaxes to equal counts.

    At every step one of the `N` numbered balls is drawn at random and moved to the other box, so the count `n` of the
    first box falls by one with probability `n / N` and rises by one otherwise. Started with all balls in the first
    box, the count relaxes to `N / 2` with the exponential mean `N / 2 + (n_0 - N / 2) (1 - 2 / N)^t`, and then
    fluctuates around it: the chain is reversible, yet the relaxation looks irreversible. Its stationary distribution
    is the binomial distribution of `N` balls with probability one half.

    Args:
        n_balls:
            The number of balls `N`.

        in_first:
            The initial number of balls in the first box, or `None` for all of them.

    Returns:
        system:
            The system, with the count `urn.n` of balls in the first box.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ehrenfest_urn
        >>> ehrenfest_urn(10).elements[0].n
        10
    """
    n = n_balls if in_first is None else in_first
    return System(elements=[EhrenfestUrnElement('urn', n=n, N=n_balls)])
