"""Self-organized criticality models: sandpiles and forest fires."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np

from ....core.elements import LatticeElement
from ....core.spaces import (
    AdjacencyNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
    TriangularNeighborhood,
)
from ....core.system import System
from ..._catalog import register_model
from ._avalanche import AvalancheInteraction
from ._fire import FireInteraction
from ._forest import ForestElement
from ._sandpile_cell import SandpileCellElement

SITE_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'triangular': TriangularNeighborhood(center=True),
    'graph': AdjacencyNeighborhood(center=True),
}


MANNA_THRESHOLD = 2


@register_model(
    field='physics/criticality',
    tags=('lattice', 'avalanches', 'power law'),
    references=('Bak, Tang, and Wiesenfeld 1987', 'Charbonneau 2017'),
    defaults={'until': 20000, 'dt': 100},
)
def sandpile(shape: Sequence[int] = (50, 50), *, random_state: int | np.random.Generator | None = 0) -> System:
    """Return the Bak-Tang-Wiesenfeld sandpile on a lattice with open edges, started from random stable heights.

    Its avalanches are recorded as `avalanche` events with their sizes, the number of topplings, their durations,
    and their areas.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

    Returns:
        system:
            The model, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.models.physics.criticality import sandpile
        >>> sandpile((10, 10)).elements[0].check().prepare().size_
        100
    """
    heights = np.random.default_rng(random_state).integers(0, 4, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[LatticeElement('pile', element=SandpileCellElement, space=space, values={'height': heights})],
        interactions=[AvalancheInteraction('avalanche', target='pile')],
    )


@register_model(
    field='physics/criticality',
    tags=('lattice', 'avalanches', 'power law', 'stochastic'),
    references=('Manna 1991', 'Jensen 2023'),
    defaults={'until': 10000, 'dt': 50},
)
def manna_sandpile(shape: Sequence[int] = (32, 32), *, random_state: int | np.random.Generator | None = 0) -> System:
    """Return the Manna sandpile, a stochastic sandpile of another universality class than Bak-Tang-Wiesenfeld.

    The pile starts from random stable heights of 0 or 1 grain, and its avalanches are recorded as `avalanche`
    events with their sizes, durations, and areas.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

    Returns:
        system:
            The model, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.models.physics.criticality import manna_sandpile
        >>> manna_sandpile((10, 10)).elements[0].check().prepare().size_
        100
    """
    heights = np.random.default_rng(random_state).integers(0, MANNA_THRESHOLD, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[LatticeElement('pile', element=SandpileCellElement, space=space, values={'height': heights})],
        interactions=[AvalancheInteraction('avalanche', target='pile', threshold=MANNA_THRESHOLD, mode='random')],
    )


@register_model(
    field='physics/criticality',
    tags=('lattice', 'fire', 'stochastic', 'power law'),
    references=('Drossel and Schwabl 1992', 'Charbonneau 2017'),
    defaults={'until': 500, 'dt': 1},
)
def forest_fire(
    shape: Sequence[int] = (100, 100),
    *,
    growth: float = 0.05,
    lightning: float = 1e-4,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Drossel-Schwabl forest fire, where slow growth and rare lightning give fires of every size.

    Args:
        shape:
            The number of rows and columns.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

        density:
            The initial fraction of sites with a tree.

        random_state:
            The seed or generator of the initial forest.

    Returns:
        system:
            The model, with the covers `forest.cover`.

    Examples:
        >>> from skcomplex.models.physics.criticality import forest_fire
        >>> forest_fire((10, 10)).elements[0].values['growth']
        0.05
    """
    cover = (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)
    space = LatticeSpace(tuple(shape))
    return System(
        elements=[
            LatticeElement(
                'forest',
                element=ForestElement,
                space=space,
                values={'cover': cover, 'growth': growth, 'lightning': lightning},
            ),
        ],
        interactions=[FireInteraction('fire', target='forest')],
    )


@register_model(
    field='physics/criticality',
    tags=('lattice', 'network', 'fire', 'stochastic', 'power law'),
    references=('Drossel and Schwabl 1992', 'Argun et al. 2021'),
    defaults={'until': 400, 'dt': 2},
)
def hexagonal_forest_fire(
    shape: Sequence[int] = (80, 80),
    *,
    graph: nx.Graph | None = None,
    growth: float = 0.02,
    lightning: float = 1e-5,
    density: float = 0.5,
    cover: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Drossel-Schwabl forest fire on a triangular lattice of hexagonal cells, or on the nodes of a graph.

    Every site of the periodic triangular lattice has six neighbours, and every node of a graph has the neighbours
    along its edges. Slow growth and rare lightning give fires of every size.

    Args:
        shape:
            The number of rows and columns of the triangular lattice, unused on a graph.

        graph:
            The graph whose nodes are the sites, or `None` for the triangular lattice.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

        density:
            The initial fraction of sites with a tree.

        cover:
            The initial cover of every site, 0 for empty, 1 for a tree, and 2 for burning, in lattice shape or node
            order, or `None` for random trees at the given density.

        random_state:
            The seed or generator of the initial forest.

    Returns:
        system:
            The model, with the covers `forest.cover`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.physics.criticality import hexagonal_forest_fire
        >>> hexagonal_forest_fire((10, 10)).elements[0].check().prepare().size_
        100
        >>> hexagonal_forest_fire(graph=nx.cycle_graph(12)).elements[0].check().prepare().size_
        12
    """
    space = (
        (LatticeSpace(tuple(shape), geometry='triangular') if graph is None else NetworkSpace(graph)).check().prepare()
    )
    size = int(space.size_ or 0)
    initial = (
        (np.random.default_rng(random_state).random(size) < density).astype(float)
        if cover is None
        else np.ravel(np.asarray(cover, dtype=float))
    )
    forest = LatticeElement(
        'forest',
        element=ForestElement,
        space=space,
        values={'cover': initial, 'growth': growth, 'lightning': lightning},
    )
    neighborhood = SITE_NEIGHBORHOODS['triangular' if graph is None else 'graph']
    return System(elements=[forest], interactions=[FireInteraction('fire', target='forest', neighborhood=neighborhood)])
