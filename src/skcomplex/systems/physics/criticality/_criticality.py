"""Self-organized criticality systems: sandpiles and forest fires."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import CascadeInteraction, LocalRuleInteraction
from ....core.components.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
    TriangularNeighborhood,
    VonNeumannNeighborhood,
)
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._sandpile_cell import SandpileCellElement


class ForestElement(Element):
    """Site of the Drossel-Schwabl forest fire: empty at 0, a tree at 1, or burning at 2.

    A `FireInteraction` grows trees on empty sites with probability `growth`, sets them on fire from a burning
    neighbour or from lightning with probability `lightning`, and empties burning sites.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        cover:
            The initial cover, 0, 1, or 2.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ForestElement
        >>> ForestElement('site').growth
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cover: Annotated[float, StateSpec(domain=NN)] = 0,
        growth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        lightning: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cover = cover
        self.growth = growth
        self.lightning = lightning


class AvalancheInteraction(CascadeInteraction):
    """Slowly driven sandpile: every step drops one grain on a random site, and the full sites topple in avalanches.

    A site that holds `threshold` grains topples. In the mode `'redistribute'` it sends one grain to each of its
    neighbours, as in the Bak-Tang-Wiesenfeld sandpile, and in the mode `'random'` every grain goes to a neighbour drawn
    at random, as in the Manna sandpile. Grains that topple over a fixed edge leave the pile, and every avalanche is
    recorded as an event under the label of the interaction.

    Args:
        label:
            The label of the interaction and of its avalanche events.

        target:
            The path of the lattice group of sites.

        state:
            The name of the state that holds the grains.

        threshold:
            The number of grains at which a site topples.

        mode:
            `'redistribute'` or `'random'`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import AvalancheInteraction
        >>> AvalancheInteraction('avalanche', target='pile', threshold=2, mode='random').share
        1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'height',
        threshold: int = 4,
        mode: str = 'redistribute',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            threshold=threshold,
            share=1,
            mode=mode,
            drive=self._drop_grain,
        )

    @staticmethod
    def _drop_grain(heights: Array, *, random: np.random.Generator) -> Array:
        """Return the heights with one grain added on a random site."""
        following = heights.copy()
        following[tuple(random.integers(size) for size in heights.shape)] += 1
        return following


NEAREST = VonNeumannNeighborhood(center=True)


BURNING = 2


class FireInteraction(LocalRuleInteraction):
    """Stochastic rule of the Drossel-Schwabl forest fire on a cover of 0 for empty, 1 for a tree, and 2 for burning.

    Every step a tree catches fire from a burning neighbour or from lightning with the probability `lightning` of the
    site, a burning site becomes empty, and a tree grows on an empty site with the probability `growth` of the site.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the lattice or network group of sites.

        neighborhood:
            The neighbours a fire spreads to, the von Neumann neighbourhood of a square lattice, the six neighbours of
            a triangular lattice, or the neighbours of a node, each with the site itself.

    Examples:
        >>> from skcomplex.systems.physics.criticality import FireInteraction
        >>> FireInteraction('fire', target='forest').stochastic
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        neighborhood: BaseNeighborhood = NEAREST,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='cover',
            rule=self._burn,
            neighborhood=neighborhood,
            stochastic=True,
        )

    @staticmethod
    def _burn(cover: Array, *around: Array, growth: float | Array, lightning: float | Array, random: Array) -> Array:
        """Return the covers after trees catch fire, fires burn out, and trees grow."""
        burning = np.any(np.stack(around) == BURNING, axis=0) if around else np.zeros(np.shape(cover), dtype=bool)
        ignites = burning | (random < lightning)
        tree = np.where(ignites, float(BURNING), 1.0)
        empty = np.where(random < growth, 1.0, 0.0)
        return np.where(cover == BURNING, 0.0, np.where(cover == 1, tree, empty))


SITE_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'triangular': TriangularNeighborhood(center=True),
    'graph': AdjacencyNeighborhood(center=True),
}


MANNA_THRESHOLD = 2


@register_system(
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
            The system, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import sandpile
        >>> sandpile((10, 10)).elements[0].prepare().size_
        100
    """
    heights = np.random.default_rng(random_state).integers(0, 4, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[LatticeElement('pile', element=SandpileCellElement, space=space, values={'height': heights})],
        interactions=[AvalancheInteraction('avalanche', target='pile')],
    )


@register_system(
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
            The system, with the heights `pile.height`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import manna_sandpile
        >>> manna_sandpile((10, 10)).elements[0].prepare().size_
        100
    """
    heights = np.random.default_rng(random_state).integers(0, MANNA_THRESHOLD, size=tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='fixed')
    return System(
        elements=[LatticeElement('pile', element=SandpileCellElement, space=space, values={'height': heights})],
        interactions=[AvalancheInteraction('avalanche', target='pile', threshold=MANNA_THRESHOLD, mode='random')],
    )


@register_system(
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
            The system, with the covers `forest.cover`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import forest_fire
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


@register_system(
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
            The system, with the covers `forest.cover`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.physics.criticality import hexagonal_forest_fire
        >>> hexagonal_forest_fire((10, 10)).elements[0].prepare().size_
        100
        >>> hexagonal_forest_fire(graph=nx.cycle_graph(12)).elements[0].prepare().size_
        12
    """
    space = (LatticeSpace(tuple(shape), geometry='triangular') if graph is None else NetworkSpace(graph)).prepare()
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
