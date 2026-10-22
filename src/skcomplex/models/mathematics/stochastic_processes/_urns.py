"""Stochastic process models: urns with reinforcement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, state
from ....core.components import Element
from ....core.laws import Jump, Law, Symbols
from ....core.runtime import register_model
from ....core.system import System


class PolyaUrn(Element):
    """Urn of red and blue balls, from which a drawn ball returns with extra balls of its color.

    Draws happen at rate 1, and a draw picks a color with probability proportional to its number of balls, so the
    sequence of draws is that of Pólya's urn in discrete time.

    Args:
        label:
            The label of the urn or the group.

        red:
            The initial number of red balls.

        blue:
            The initial number of blue balls.

        added:
            The number of balls of the drawn color added after every draw.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import PolyaUrn
        >>> PolyaUrn('urn', added=2).added
        2
    """

    red: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    blue: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    added: int = 1

    def laws(self, s: Symbols) -> list[Law]:
        """Return the draws of a red and of a blue ball.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        total = s.red + s.blue
        return [Jump(s.red / total, {s.red: self.added}), Jump(s.blue / total, {s.blue: self.added})]


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
        >>> polya_urn(3).elements[0].size
        3
    """
    return System(components=[PolyaUrn.group('urns', n_urns, red=float(red), blue=float(blue), added=added)])
