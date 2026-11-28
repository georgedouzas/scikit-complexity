"""Self-organized criticality systems.

The module covers sandpiles and forest fires.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    LatticeSpace,
    NetworkSpace,
    RadiusNeighborhood,
)
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.avalanches import TopplingInteraction

NEAREST = RadiusNeighborhood(1, metric='manhattan', center=True)
BURNING = 2
SITE_NEIGHBORHOODS: dict[str, RadiusNeighborhood | AdjacencyNeighborhood] = {
    'triangular': RadiusNeighborhood(1, metric='hexagonal', center=True),
    'graph': AdjacencyNeighborhood(center=True),
}
MANNA_THRESHOLD = 2
FOREST_BOUNDARIES = ('periodic', 'fixed', 'reflecting')


class ForestElement(Element):
    """Sites of the Drossel-Schwabl forest fire.

    Every site is empty at 0, a tree at 1, or burning at 2.

    A `FireInteraction` grows trees on empty sites with probability `growth`, sets them on fire from a burning
    neighbour or from lightning with probability `lightning`, and empties burning sites.

    Args:
        label:
            The label of the site or the lattice of sites.

        cover:
            The initial cover, 0, 1, or 2, or one cover per site.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

        space:
            The lattice or network of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ForestElement
        >>> ForestElement('site').growth
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        cover: object = 0.0,
        growth: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        lightning: Annotated[float, RealSet.closed_open(0, oo)] = 0.0001,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.cover = cover
        self.growth = growth
        self.lightning = lightning


class AvalancheInteraction(TopplingInteraction):
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
        target: Role,
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
    def _drop_grain(heights: Array, *, random: np.random.Generator, **kwargs: object) -> Array:
        """Return the heights with one grain added on a random site."""
        following = heights.copy()
        following[tuple(random.integers(size) for size in heights.shape)] += 1
        return following


class FireInteraction(Interaction):
    """Stochastic rule of the Drossel-Schwabl forest fire.

    The cover is 0 for empty, 1 for a tree, and 2 for burning.

    Every step a tree catches fire from a burning neighbour or from lightning with the probability `lightning` of the
    site, a burning site becomes empty, and a tree grows on an empty site with the probability `growth` of the site.
    The rule, labelled `fire`, records every fire as a `fire` event, the burning sites connected in space and time.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the lattice or network of sites.

        neighborhood:
            The neighbours a fire spreads to, the von Neumann neighbourhood of a square lattice, the six neighbours of
            a triangular lattice, or the neighbours of a node, each with the site itself.

    Examples:
        >>> from skcomplex.systems.physics.criticality import FireInteraction
        >>> FireInteraction('fire', target='forest').neighborhood.metric
        'manhattan'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, neighborhood: BaseNeighborhood = NEAREST) -> None:
        super().__init__(label)
        self.target = target
        self.neighborhood = neighborhood

    @staticmethod
    def _burn(
        cover: Array,
        *around: Array,
        growth: float | Array,
        lightning: float | Array,
        random: Array,
        **kwargs: object,
    ) -> Array:
        """Return the covers after trees catch fire, fires burn out, and trees grow."""
        burning = np.any(np.stack(around) == BURNING, axis=0) if around else np.zeros(np.shape(cover), dtype=bool)
        ignites = burning | (random < lightning)
        tree = np.where(ignites, float(BURNING), 1.0)
        empty = np.where(random < growth, 1.0, 0.0)
        return np.where(cover == BURNING, 0.0, np.where(cover == 1, tree, empty))

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule of the covers of the target.

        Args:
            context:
                The namespace of the target and the time `t`.
        """
        super()._prepare(context)
        cover = context.target.cover
        self.laws_ = [
            UpdateLaw(
                cover,
                self._burn,
                neighborhood=self.neighborhood,
                stochastic=True,
                activity=where(cover == BURNING, 1, 0),
                label='fire',
            ),
        ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SandpileSystem
        >>> trajectory = SandpileSystem((10, 10)).prepare().simulate(until=2, dt=1, random_state=0)
        >>> trajectory.get_values('pile.height').shape
        (3, 100)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Bak-Tang-Wiesenfeld sandpile, which starts from random stable heights.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        heights = np.random.default_rng(self.random_state).integers(0, 4, size=tuple(self.shape)).astype(float)
        space = LatticeSpace(tuple(self.shape), boundary='fixed')
        self.system_ = System(
            self.label,
            elements=[Element('pile', values={'height': heights.ravel()}, space=space)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import MannaSandpileSystem
        >>> trajectory = MannaSandpileSystem((10, 10)).prepare().simulate(until=2, dt=1, random_state=0)
        >>> trajectory.get_values('pile.height').shape
        (3, 100)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Manna sandpile, a stochastic sandpile of another universality class than Bak-Tang-Wiesenfeld.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        heights = (
            np.random.default_rng(self.random_state).integers(0, MANNA_THRESHOLD, size=tuple(self.shape)).astype(float)
        )
        space = LatticeSpace(tuple(self.shape), boundary='fixed')
        self.system_ = System(
            self.label,
            elements=[Element('pile', values={'height': heights.ravel()}, space=space)],
            interactions=[AvalancheInteraction('avalanche', target='pile', threshold=MANNA_THRESHOLD, mode='random')],
        )


class ForestFireSystem(BaseSystem):
    """The Drossel-Schwabl forest fire.

    Slow growth and rare lightning give fires of every size.

    Every fire is recorded as a `fire` event, the burning sites connected in space and time, with its size, the number
    of trees that burn, its duration, its area, and its profile, the burning sites of every step. Two fires that touch
    are one fire, the usual cluster of burning sites of a forest fire model, and a fire still burning when the run ends
    is left out. The system has the covers `forest.cover`.

    Args:
        shape:
            The number of rows and columns.

        boundary:
            The edges of the forest. `'periodic'` joins opposite edges, so a fire at one edge reaches the opposite edge
            in one step. `'fixed'` holds no trees beyond the edges, so no fire crosses them, and `'reflecting'` reads
            the mirror image of the forest beyond them.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The boundary is none of `'periodic'`, `'fixed'`, and `'reflecting'`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ForestFireSystem
        >>> ForestFireSystem((10, 10)).prepare().system_.elements[0].growth
        0.05
        >>> trajectory = ForestFireSystem((32, 32), lightning=1e-3).prepare().simulate(until=100, random_state=0)
        >>> {event.label for event in trajectory.events}
        {'fire'}
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
        boundary: str = 'periodic',
        growth: float = 0.05,
        lightning: float = 1e-4,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.boundary = boundary
        self.growth = growth
        self.lightning = lightning
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Drossel-Schwabl forest fire, where slow growth and rare lightning give fires of every size.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The boundary is none of `'periodic'`, `'fixed'`, and `'reflecting'`.
        """
        super()._prepare(context)
        if self.boundary not in FOREST_BOUNDARIES:
            error_msg = f'ForestFireSystem has the boundary {self.boundary!r}. Use one of {", ".join(FOREST_BOUNDARIES)}.'
            raise ValidationError(error_msg)
        cover = (np.random.default_rng(self.random_state).random(tuple(self.shape)) < self.density).astype(float)
        forest = ForestElement(
            'forest',
            cover=cover.ravel(),
            growth=self.growth,
            lightning=self.lightning,
            space=LatticeSpace(tuple(self.shape), boundary=self.boundary),
        )
        self.system_ = System(self.label, elements=[forest], interactions=[FireInteraction('fire', target='forest')])


class HexagonalForestFireSystem(BaseSystem):
    """The Drossel-Schwabl forest fire on a hexagonal lattice or on a graph.

    The hexagonal cells form a triangular lattice, and on a graph the sites are its nodes.

    Every site of the periodic triangular lattice has six neighbours, and every node of a graph has the neighbours along
    its edges. Slow growth and rare lightning give fires of every size. Every fire is recorded as a `fire` event, the
    burning sites connected in space and time along the neighbours of the cells or the edges of the graph, whose size
    is the number of trees that burn. The system has the covers `forest.cover`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.physics.criticality import HexagonalForestFireSystem
        >>> HexagonalForestFireSystem((10, 10)).prepare().system_.elements[0].space.prepare().size_
        100
        >>> HexagonalForestFireSystem(graph=nx.cycle_graph(12)).prepare().system_.elements[0].space.prepare().size_
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.graph = graph
        self.growth = growth
        self.lightning = lightning
        self.density = density
        self.cover = cover
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the forest fire on the cells of a triangular lattice or on the nodes of a graph.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        space = (
            LatticeSpace(tuple(self.shape), geometry='triangular') if self.graph is None else NetworkSpace(self.graph)
        ).prepare()
        size = int(space.size_ or 0)
        initial = (
            (np.random.default_rng(self.random_state).random(size) < self.density).astype(float)
            if self.cover is None
            else np.ravel(np.asarray(self.cover, dtype=float))
        )
        forest = ForestElement('forest', cover=initial, growth=self.growth, lightning=self.lightning, space=space)
        neighborhood = SITE_NEIGHBORHOODS['triangular' if self.graph is None else 'graph']
        self.system_ = System(
            self.label,
            elements=[forest],
            interactions=[FireInteraction('fire', target='forest', neighborhood=neighborhood)],
        )
