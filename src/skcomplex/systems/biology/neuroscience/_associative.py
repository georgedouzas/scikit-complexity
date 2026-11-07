"""Neuroscience systems: associative thought processes as transient attractors of clique-encoding networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._associative_site import AssociativeSiteElement
from ._clique_competition import CliqueCompetitionInteraction

CLIQUES = ((0, 1), (0, 6), (3, 6), (1, 2, 3), (4, 5, 6), (1, 2, 4, 5))


@register_system(
    field='biology/neuroscience',
    tags=('associative memory', 'transient attractors', 'winners-take-all', 'cognitive systems', 'ode'),
    references=('Gros 2007', 'Gros 2015'),
    defaults={'until': 1500.0, 'dt': 2.0},
)
def associative_thought_process(
    cliques: Sequence[Sequence[int]] = CLIQUES,
    *,
    w: float = 0.15,
    z: float = 1.0,
    x_c: float = 0.85,
    gamma_plus: float = 0.005,
    gamma_minus: float = 0.005,
    start: Sequence[int] = (0, 1),
) -> System:
    """Return the dense homogeneous associative network of Gros, whose memories become active one after the other.

    Every memory is a clique of sites joined by excitatory links of strength `w`, and every other pair of sites
    inhibits each other with strength `z`. Without reservoirs the network is a winners-take-all network whose cliques
    are stable attractors. The reservoirs of an active clique deplete slowly, and once they are low the clique loses
    its excitation and its hold on the other sites, so the activity hops to an associated clique that shares sites with
    it. The memories become transient attractors, and the network runs through a sequence of them, an associative
    thought process. The default is the network of seven sites and six cliques of Gros (2015, figure 10.12).

    Args:
        cliques:
            The memories, each a sequence of site indices.

        w:
            The strength of the excitatory links within a clique.

        z:
            The strength of the inhibition between sites of no common clique.

        x_c:
            The activity above which a site counts as active.

        gamma_plus:
            The recovery rate of the reservoirs.

        gamma_minus:
            The depletion rate of the reservoirs.

        start:
            The sites that are active at the start.

    Returns:
        system:
            The system, with the activities `sites.x` and the reservoirs `sites.phi`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import associative_thought_process
        >>> associative_thought_process().elements[0].prepare().size_
        7
    """
    size = 1 + max(max(clique) for clique in cliques)
    links = np.zeros((size, size))
    for clique in cliques:
        for first, second in combinations(clique, 2):
            links[first, second] = links[second, first] = w
    activities = np.zeros(size)
    activities[list(start)] = 1.0
    sites = GroupElement(
        'sites',
        element=AssociativeSiteElement,
        size=size,
        values={'x': activities, 'phi': 1.0, 'x_c': x_c, 'gamma_plus': gamma_plus, 'gamma_minus': gamma_minus},
    )
    competition = CliqueCompetitionInteraction('competition', sites='sites', w=tuple(map(tuple, links.tolist())), z=z)
    return System(elements=[sites], interactions=[competition])
