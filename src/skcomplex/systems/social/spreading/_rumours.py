"""Rumour systems: news that spreads by contact until spreaders lose interest."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import System
from ..._catalog import register_system
from ._listeners import ListenersElement
from ._rumour_contacts import RumourContactsInteraction


@register_system(
    field='social/spreading',
    tags=('rumours', 'jump process', 'stochastic'),
    references=('Maki and Thompson 1973', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 20.0, 'dt': 0.5},
)
def maki_thompson(n_people: int = 1000, *, spread: float = 1.0, stifle: float = 1.0) -> System:
    """Return the Maki-Thompson rumour system started by one spreader, which leaves about 20% of people ignorant.

    Args:
        n_people:
            The size of the population.

        spread:
            The contact rate that spreads the rumour.

        stifle:
            The contact rate that stifles a spreader.

    Returns:
        system:
            The system, with the counts `ignorants.n`, `spreaders.n`, and `stiflers.n`.

    Examples:
        >>> from skcomplex.systems.social.spreading import maki_thompson
        >>> [element.n for element in maki_thompson(100).elements]
        [99, 1, 0]
    """
    return System(
        elements=[
            ListenersElement('ignorants', n=n_people - 1),
            ListenersElement('spreaders', n=1),
            ListenersElement('stiflers', n=0),
        ],
        interactions=[
            RumourContactsInteraction(
                'contacts',
                ignorants='ignorants',
                spreaders='spreaders',
                stiflers='stiflers',
                spread=spread,
                stifle=stifle,
            ),
        ],
    )
