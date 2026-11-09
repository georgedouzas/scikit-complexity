"""Self-organized criticality systems.

The module covers sandpiles and forest fires.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CascadeInteraction, LocalRuleInteraction
from ....core.components.spaces import (
    AdjacencyNeighborhood,
    BallNeighborhood,
    BaseNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
)
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common import SandpileCellElement


class ForestElement(Element):
    """Site of the Drossel-Schwabl forest fire.

    The site is empty at 0, a tree at 1, or burning at 2.

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
    """Slowly driven sandpile.

    Every step drops one grain on a random site, and the full sites topple in avalanches.

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


NEAREST = BallNeighborhood(metric='manhattan', center='first')


BURNING = 2


class FireInteraction(LocalRuleInteraction):
    """Stochastic rule of the Drossel-Schwabl forest fire.

    The cover is 0 for empty, 1 for a tree, and 2 for burning.

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
    'triangular': BallNeighborhood(metric='hexagonal', center='first'),
    'graph': AdjacencyNeighborhood(center=True),
}


MANNA_THRESHOLD = 2


class SandpileSystem(BaseSystem):
    """The Bak-Tang-Wiesenfeld sandpile on a lattice with open edges.

    The pile starts from random stable heights.

    Its avalanches are recorded as `avalanche` events with their sizes, the number of topplings, their durations, and
    their areas. The system has the heights `pile.height`.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SandpileSystem
        >>> SandpileSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'physics/criticality'
    topics = ('lattice', 'avalanches', 'power law')
    references = ('Bak, Tang, and Wiesenfeld 1987', 'Charbonneau 2017')
    until = 20000
    dt = 100

    def __init__(
        self,
        shape: Sequence[int] = (50, 50),
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the Bak-Tang-Wiesenfeld sandpile on a lattice with open edges.

        The pile starts from random stable heights.
        """
        heights = np.random.default_rng(self.random_state).integers(0, 4, size=tuple(self.shape)).astype(float)
        space = LatticeSpace(tuple(self.shape), boundary='fixed')
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('pile', element=SandpileCellElement, space=space, values={'height': heights}, on='sites'),
            ],
            interactions=[AvalancheInteraction('avalanche', target='pile')],
        )


class MannaSandpileSystem(BaseSystem):
    """The Manna sandpile.

    It is a stochastic sandpile of another universality class than Bak-Tang-Wiesenfeld.

    The pile starts from random stable heights of 0 or 1 grain, and its avalanches are recorded as `avalanche` events
    with their sizes, durations, and areas. The system has the heights `pile.height`.

    Args:
        shape:
            The number of rows and columns.

        random_state:
            The seed or generator of the initial heights.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import MannaSandpileSystem
        >>> MannaSandpileSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'physics/criticality'
    topics = ('lattice', 'avalanches', 'power law', 'stochastic')
    references = ('Manna 1991', 'Jensen 2023')
    until = 10000
    dt = 50

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the Manna sandpile.

        It is a stochastic sandpile of another universality class than Bak-Tang-Wiesenfeld.
        """
        heights = (
            np.random.default_rng(self.random_state).integers(0, MANNA_THRESHOLD, size=tuple(self.shape)).astype(float)
        )
        space = LatticeSpace(tuple(self.shape), boundary='fixed')
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('pile', element=SandpileCellElement, space=space, values={'height': heights}, on='sites'),
            ],
            interactions=[AvalancheInteraction('avalanche', target='pile', threshold=MANNA_THRESHOLD, mode='random')],
        )


class ForestFireSystem(BaseSystem):
    """The Drossel-Schwabl forest fire.

    Slow growth and rare lightning give fires of every size.

    The system has the covers `forest.cover`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ForestFireSystem
        >>> ForestFireSystem((10, 10)).prepare().system_.elements[0].values['growth']
        0.05
    """

    field = 'physics/criticality'
    topics = ('lattice', 'fire', 'stochastic', 'power law')
    references = ('Drossel and Schwabl 1992', 'Charbonneau 2017')
    until = 500
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        growth: float = 0.05,
        lightning: float = 1e-4,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.growth = growth
        self.lightning = lightning
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the Drossel-Schwabl forest fire.

        Slow growth and rare lightning give fires of every size.
        """
        cover = (np.random.default_rng(self.random_state).random(tuple(self.shape)) < self.density).astype(float)
        space = LatticeSpace(tuple(self.shape))
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'forest',
                    element=ForestElement,
                    space=space,
                    values={'cover': cover, 'growth': self.growth, 'lightning': self.lightning},
                    on='sites',
                ),
            ],
            interactions=[FireInteraction('fire', target='forest')],
        )


class HexagonalForestFireSystem(BaseSystem):
    """The Drossel-Schwabl forest fire on a hexagonal lattice or on a graph.

    The hexagonal cells form a triangular lattice, and on a graph the sites are its nodes.

    Every site of the periodic triangular lattice has six neighbours, and every node of a graph has the neighbours along
    its edges. Slow growth and rare lightning give fires of every size. The system has the covers `forest.cover`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.physics.criticality import HexagonalForestFireSystem
        >>> HexagonalForestFireSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
        >>> HexagonalForestFireSystem(graph=nx.cycle_graph(12)).prepare().system_.elements[0].prepare().size_
        12
    """

    field = 'physics/criticality'
    topics = ('lattice', 'network', 'fire', 'stochastic', 'power law')
    references = ('Drossel and Schwabl 1992', 'Argun et al. 2021')
    until = 400
    dt = 2

    def __init__(
        self,
        shape: Sequence[int] = (80, 80),
        *,
        graph: nx.Graph | None = None,
        growth: float = 0.02,
        lightning: float = 1e-5,
        density: float = 0.5,
        cover: Sequence[float] | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.graph = graph
        self.growth = growth
        self.lightning = lightning
        self.density = density
        self.cover = cover
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        space = (
            LatticeSpace(tuple(self.shape), geometry='triangular') if self.graph is None else NetworkSpace(self.graph)
        ).prepare()
        size = int(space.size_ or 0)
        initial = (
            (np.random.default_rng(self.random_state).random(size) < self.density).astype(float)
            if self.cover is None
            else np.ravel(np.asarray(self.cover, dtype=float))
        )
        forest = GroupElement(
            'forest',
            element=ForestElement,
            space=space,
            values={'cover': initial, 'growth': self.growth, 'lightning': self.lightning},
            on='sites',
        )
        neighborhood = SITE_NEIGHBORHOODS['triangular' if self.graph is None else 'graph']
        return CompositeSystem(
            self.label,
            elements=[forest],
            interactions=[FireInteraction('fire', target='forest', neighborhood=neighborhood)],
        )
