"""Variants of self-organized criticality.

The module covers the fixed-energy sandpile, the Zhang sandpile, and branching avalanches.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.laws.utils import draw, either, where
from ....core.components.spaces import BallNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ObservableSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.avalanches import SandpileCellElement, TopplingInteraction

THRESHOLD = 4


class EnergySiteElement(Element):
    """Site of the Zhang sandpile.

    The site stores a continuous energy.

    A cascade loads the energy in random packets and resets the sites above the threshold.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        energy:
            The initial energy.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import EnergySiteElement
        >>> EnergySiteElement('site').energy
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        energy: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy


class ToppleInteraction(LocalRuleInteraction):
    """Parallel toppling of a sandpile without drive.

    Every site at or above the threshold topples once a step.

    Every step, every site that holds at least `threshold` grains loses `threshold` grains, and every site gains one
    grain from each toppling neighbour, all at once. On a periodic lattice no grain is lost, so the number of grains is
    conserved, and the activity either dies out in an absorbing configuration or lasts forever.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the lattice group of sites.

        threshold:
            The number of grains at which a site topples, the number of its neighbours.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ToppleInteraction
        >>> ToppleInteraction('topple', target='pile').threshold
        4
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], threshold: int = 4) -> None:
        super().__init__(
            label,
            target=target,
            state='height',
            rule=self._get_topple,
            neighborhood=BallNeighborhood(metric='manhattan', center='first'),
        )
        self.threshold = threshold

    def _get_topple(self, height: Array, *around: Array) -> Array:
        """Return the heights after every site at or above the threshold topples once."""
        gained = np.sum([neighbor >= self.threshold for neighbor in around], axis=0)
        return height - self.threshold * (height >= self.threshold) + gained


def _get_active(site: SimpleNamespace) -> Expression:
    """Return 1 for an active site and 0 otherwise.

    An active site holds at least the threshold.
    """
    return where(site.height >= THRESHOLD, 1, 0)


class FixedEnergySandpileSystem(BaseSystem):
    """The fixed-energy sandpile.

    It is a Bak-Tang-Wiesenfeld pile with no drive and no loss on a periodic lattice.

    Every step, every site with at least four grains gives one grain to each of its four neighbours, all at once. The
    lattice is periodic, so the density of grains `density` is conserved, and it controls an absorbing phase transition.
    Below the critical density, about 2.125 on the square lattice, the topplings stop in an absorbing configuration, and
    above it the pile stays active forever. The density of active sites, the observable `activity`, is the order
    parameter. The grains start scattered uniformly at random over the sites. The system has the heights `pile.height`
    and the observable `activity`, the density of active sites.

    Args:
        shape:
            The number of rows and columns.

        density:
            The mean number of grains per site.

        random_state:
            The seed or generator of the initial grains.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import FixedEnergySandpileSystem
        >>> FixedEnergySandpileSystem((8, 8), density=2.0).prepare().system_.elements[0].prepare().size_
        64
    """

    field = 'physics/criticality'
    topics = ('lattice', 'absorbing states', 'phase transition', 'conservation', 'sandpile')
    references = ('Dickman, Vespignani, and Zapperi 1998', 'Vespignani et al. 2000', 'Gros 2015')
    until = 500
    dt = 1
    observables = (ObservableSpec('activity', _get_active, group='pile'),)

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        density: float = 2.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        size = int(np.prod(self.shape))
        rng = np.random.default_rng(self.random_state)
        grains = rng.integers(size, size=round(self.density * size))
        heights = np.bincount(grains, minlength=size).reshape(tuple(self.shape)).astype(float)
        space = LatticeSpace(tuple(self.shape), boundary='periodic')
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('pile', element=SandpileCellElement, space=space, values={'height': heights}, on='sites'),
            ],
            interactions=[ToppleInteraction('topple', target='pile', threshold=THRESHOLD)],
        )


class ZhangSandpileSystem(BaseSystem):
    """The Zhang sandpile.

    It is a continuous sandpile whose unstable sites give away all their energy.

    Every step loads a site chosen at random with a packet of energy of uniform size in `[0, threshold]`. A site whose
    energy reaches the threshold relaxes: its energy falls to 0 and each of its four neighbours gains a quarter of it,
    all at once, until no site is unstable. Energy that would go beyond the open edges is lost. The pile organizes
    itself into a critical state with avalanches of every size, recorded as `avalanche` events with their sizes,
    durations, and areas, and the energies of the sites cluster near a few values. The energies start uniformly at
    random below the threshold. The system has the energies `pile.energy`.

    Args:
        shape:
            The number of rows and columns.

        threshold:
            The energy at which a site relaxes, which is also the largest packet.

        random_state:
            The seed or generator of the initial energies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ZhangSandpileSystem
        >>> ZhangSandpileSystem((8, 8)).prepare().system_.interactions[0].mode
        'reset'
    """

    field = 'physics/criticality'
    topics = ('lattice', 'avalanches', 'power law', 'continuous energy', 'sandpile')
    references = ('Zhang 1989', 'Jensen 2023')
    until = 10000
    dt = 100

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        threshold: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.threshold = threshold
        self.random_state = random_state

    @staticmethod
    def _load(packet: float, energies: Array, *, random: np.random.Generator) -> Array:
        """Return the energies with a packet added on a random site.

        The packet has a uniform size in `[0, packet]`.
        """
        rng = random
        following = energies.copy()
        following[tuple(rng.integers(size) for size in energies.shape)] += rng.uniform(0.0, packet)
        return following

    def _build_system(self) -> CompositeSystem:
        """Return the Zhang sandpile.

        It is a continuous sandpile whose unstable sites give away all their energy.
        """
        energies = np.random.default_rng(self.random_state).uniform(0.0, self.threshold, size=tuple(self.shape))
        space = LatticeSpace(tuple(self.shape), boundary='fixed')
        avalanche = TopplingInteraction(
            'avalanche',
            target='pile',
            state='energy',
            threshold=self.threshold,
            share=1 / 4,
            mode='reset',
            drive=partial(self._load, self.threshold),
        )
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement('pile', element=EnergySiteElement, space=space, values={'energy': energies}, on='sites'),
            ],
            interactions=[avalanche],
        )


class SelfOrganizedBranchingSystem(DynamicalSystem):
    """The self-organized branching process of Zapperi, Lauritsen, and Stanley.

    It is a mean-field sandpile.

    An avalanche starts from one active site at the top of a binary tree of `generations` generations. Every active site
    relaxes with the probability `p` into two active sites of the next generation, and stays still otherwise, so every
    generation has `2 Binomial(z, p)` active sites. The `sigma` active sites that reach the last generation leave the
    system as dissipation. After every avalanche one unit of energy has entered and `sigma` have left, so the
    probability changes by `(1 - sigma) / N` with `N = 2^(generations + 1) - 1` sites. A supercritical `p` makes large
    dissipating avalanches that lower it, and a subcritical one lets energy accumulate, so `p` organizes itself to the
    critical value `p_c = 1 / 2`, where the branching ratio `2 p` is 1. The process advances one generation per step, so
    a step is a time of the avalanche and avalanches follow each other without pause. The system has the probability
    `tree.p`, the active sites `tree.z`, and the generation `tree.g`.

    Args:
        p:
            The initial branching probability.

        generations:
            The number of generations of the tree, at least 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SelfOrganizedBranchingSystem
        >>> SelfOrganizedBranchingSystem(0.4).prepare().system_.elements[0].get_value('p')
        0.4
    """

    draws = True

    field = 'physics/criticality'
    topics = ('branching process', 'avalanches', 'mean field', 'self-organized criticality', 'stochastic')
    references = ('Zapperi, Lauritsen, and Stanley 1995', 'Jensen 2023')
    until = 30000
    dt = 1

    @staticmethod
    def _get_ends(tree: SimpleNamespace, generations: int) -> Expression:
        """Return whether the avalanche of a tree ends this step.

        It ends at the last generation or with no active site.
        """
        return either(tree.g >= generations, tree.z == 0)

    def __init__(
        self,
        p: float = 0.3,
        *,
        generations: int = 10,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.p = p
        self.generations = generations

    @staticmethod
    def _get_branch(generations: int, s: SimpleNamespace) -> Expression:
        """Return the active sites of the next generation, or a new avalanche's one site."""
        return where(
            SelfOrganizedBranchingSystem._get_ends(s.target, generations),
            1,
            2 * draw('children', 'binom', n=s.target.z, p=s.target.p),
        )

    @staticmethod
    def _advance(generations: int, s: SimpleNamespace) -> Expression:
        """Return the next generation, or 0 for a new avalanche."""
        return where(SelfOrganizedBranchingSystem._get_ends(s.target, generations), 0, s.target.g + 1)

    @staticmethod
    def _tune(generations: int, s: SimpleNamespace) -> Expression:
        """Return the branching probability.

        It changes by `(1 - sigma) / N` at the end of an avalanche.
        """
        sites = 2 ** (generations + 1) - 1
        dissipated = where(s.target.g >= generations, s.target.z, 0)
        return where(
            SelfOrganizedBranchingSystem._get_ends(s.target, generations),
            s.target.p + (1 - dissipated) / sites,
            s.target.p,
        )

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'tree'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'p': DeclaredVariableSpec(self.p, StateSpec(domain=RealSet.closed(0, 1))),
            'z': DeclaredVariableSpec(1, StateSpec(domain=NN)),
            'g': DeclaredVariableSpec(0, StateSpec(domain=NN)),
        }

    def _get_equations(self) -> Equations:
        """Return the self-organized branching process of Zapperi, Lauritsen, and Stanley."""
        return {
            'z': partial(self._get_branch, self.generations),
            'g': partial(self._advance, self.generations),
            'p': partial(self._tune, self.generations),
        }
