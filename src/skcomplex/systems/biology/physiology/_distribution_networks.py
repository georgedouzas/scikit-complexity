"""Physiology systems: fractal distribution networks and the metabolic scaling they imply."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import NetworkSpace, StaticTime
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._branching_vessel import BranchingVesselElement


def _network_totals(system: System) -> list[ObservableSpec]:
    """Return the blood volume and the number of capillaries of the network."""
    return [
        ObservableSpec('blood_volume', group='vessels', variable='volume', reduce='sum'),
        ObservableSpec('capillaries', group='vessels', variable='terminal', reduce='sum'),
    ]


@register_system(
    field='biology/physiology',
    tags=('allometry', 'kleiber law', 'fractal network', 'static', 'metabolic scaling'),
    references=('West, Brown, and Enquist 1997', 'Ribeiro and Netto 2024'),
    observables=_network_totals,
)
def west_brown_enquist(depth: int = 6, *, n: int = 2, r_c: float = 1.0, l_c: float = 1.0) -> System:
    """Return the distribution network of West, Brown, and Enquist, a space-filling tree of invariant capillaries.

    The tree branches `n` ways at every level down to the capillaries at level `depth`. The metabolic rate is
    proportional to the number of capillaries, `n^depth`, and the body mass to the blood volume, which the
    area-preserving and space-filling vessels make `V = pi r_c^2 l_c n^depth (n^(depth / 3) - 1) / (n^(1 / 3) - 1)`,
    which grows as `n^(4 depth / 3)` for deep trees. So the metabolic rate scales as the mass to the power 3/4,
    Kleiber's law, which the exponent between consecutive depths approaches from below. An `Experiment` over `depth`
    fits it.

    Args:
        depth:
            The level of the capillaries, the number of branchings.

        n:
            The branching ratio.

        r_c:
            The radius of a capillary.

        l_c:
            The length of a capillary.

    Returns:
        system:
            The system, with the radii `vessels.r`, lengths `vessels.l`, and volumes `vessels.volume` on the edges of
            the tree and the observables `blood_volume` and `capillaries` of the catalog, which a run passes in
            `observables`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import west_brown_enquist
        >>> west_brown_enquist(3).elements[0].prepare().size_
        14
    """
    graph = nx.balanced_tree(n, depth)
    space = NetworkSpace(graph).prepare()
    _, children = space.edges_[None]
    levels = nx.single_source_shortest_path_length(graph, 0)
    vessels = GroupElement(
        'vessels',
        element=BranchingVesselElement,
        space=space,
        on='edges',
        values={
            'level': np.array([levels[int(child)] for child in children], dtype=float),
            'depth': float(depth),
            'n': float(n),
            'r_c': r_c,
            'l_c': l_c,
        },
    )
    return System(elements=[vessels], space=space, time=StaticTime())
