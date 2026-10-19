"""Rumour models: news that spreads by contact until spreaders lose interest."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Jump, Law, Symbols, System
from ....core.runtime import register_model


class Listeners(Element):
    """Number of people in one state of a rumour: ignorant, spreading, or stifling.

    Args:
        label:
            The label of the element.

        n:
            The initial number of people.

    Examples:
        >>> from skcomplex.models.social.spreading import Listeners
        >>> Listeners('spreaders', n=1).n
        1
    """

    n: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)


class RumourContacts(Interaction):
    """Contacts of the Maki-Thompson rumour model in a well-mixed population of size `N`.

    A spreader who meets an ignorant tells the rumour at rate `spread * X * Y / N`, and a spreader who meets another
    spreader or a stifler stops spreading at rate `stifle * Y * (Y - 1 + Z) / N`.

    Args:
        label:
            The label of the interaction.

        ignorants:
            The path of the ignorants `X`.

        spreaders:
            The path of the spreaders `Y`.

        stiflers:
            The path of the stiflers `Z`.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

    Examples:
        >>> from skcomplex.models.social.spreading import RumourContacts
        >>> RumourContacts('contacts', ignorants='X', spreaders='Y', stiflers='Z').spread
        1.0
    """

    ignorants: str = role()
    spreaders: str = role()
    stiflers: str = role()
    spread: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    stifle: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the telling and stifling jumps.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        ignorants, spreaders, stiflers = s.ignorants.n, s.spreaders.n, s.stiflers.n
        total = ignorants + spreaders + stiflers
        return [
            Jump(s.spread * ignorants * spreaders / total, {ignorants: -1, spreaders: 1}),
            Jump(s.stifle * spreaders * (spreaders - 1 + stiflers) / total, {spreaders: -1, stiflers: 1}),
        ]


@register_model(
    field='social/spreading',
    tags=('rumours', 'jump process', 'stochastic'),
    references=('Maki and Thompson 1973', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 20.0, 'dt': 0.5},
)
def maki_thompson(n_people: int = 1000, *, spread: float = 1.0, stifle: float = 1.0) -> System:
    """Return the Maki-Thompson rumour model started by one spreader, which leaves about 20% of people ignorant.

    Args:
        n_people:
            The size of the population.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

    Returns:
        system:
            The model, with the counts `ignorants.n`, `spreaders.n`, and `stiflers.n`.

    Examples:
        >>> from skcomplex.models.social.spreading import maki_thompson
        >>> [element.n for element in maki_thompson(100).elements]
        [99, 1, 0]
    """
    return System(
        elements=[Listeners('ignorants', n=n_people - 1), Listeners('spreaders', n=1), Listeners('stiflers', n=0)],
        interactions=[
            RumourContacts(
                'contacts',
                ignorants='ignorants',
                spreaders='spreaders',
                stiflers='stiflers',
                spread=spread,
                stifle=stifle,
            ),
        ],
    )
