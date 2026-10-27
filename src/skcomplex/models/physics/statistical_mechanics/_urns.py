"""Urn models: the Ehrenfest model of balls moved one at a time between two boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import UpdateLaw, bernoulli
from ....core.system import System
from ....engine.runtime import register_model
from ...core.base import BaseLaw
from ...core.laws import Symbols


class EhrenfestUrnElement(Element):
    """Ehrenfest urn element."""

    n = state(default=0, domain=NumericDomain.NONNEGATIVE_INTEGER)
    N = parameter(40, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the ehrenfest urn element."""
        return [UpdateLaw(s.n, s.n + 1 - 2 * bernoulli('pick', s.n / s.N))]


@register_model(
    field='physics/statistical_mechanics',
    tags=('markov chain', 'irreversibility', 'entropy', 'discrete map'),
    references=('Ehrenfest and Ehrenfest 1907', 'Barsky 2020', 'Kac 1947'),
    defaults={'until': 200, 'dt': 1},
)
def ehrenfest_urn(n_balls: int = 40, *, in_first: int | None = None) -> System:
    """Return the Ehrenfest urn model, balls moved one at a time between two boxes, which relaxes to equal counts.

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
            The model, with the count `urn.n` of balls in the first box.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ehrenfest_urn
        >>> ehrenfest_urn(10).elements[0].n
        10
    """
    n = n_balls if in_first is None else in_first
    return System(elements=[EhrenfestUrnElement('urn', n=n, N=n_balls)])
