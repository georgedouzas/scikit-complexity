"""Variants of self-organized criticality.

The module covers the fixed-energy sandpile, the Zhang sandpile, and branching avalanches.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import draw, either, where
from ....core.components.spaces import BaseSpace, LatticeSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.avalanches import TopplingInteraction

THRESHOLD = 4


class EnergySiteElement(Element):
    """Sites of the Zhang sandpile.

    Every site stores a continuous energy.

    A cascade loads the energy in random packets and resets the sites above the threshold.

    Args:
        label:
            The label of the site or the lattice of sites.

        energy:
            The initial energy, or one value per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.criticality import EnergySiteElement
        >>> EnergySiteElement('site').energy
        0.0
    """

    def __init__(self, label: str, *, energy: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.energy = energy


class GrainSiteElement(Element):
    """Sites of a sandpile.

    Every site holds a number of grains.

    Args:
        label:
            The label of the site or the lattice of sites.

        height:
            The initial number of grains, or one number per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.criticality import GrainSiteElement
        >>> GrainSiteElement('site').height
        0.0
    """

    def __init__(self, label: str, *, height: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.height = height


class ToppleInteraction(Interaction):
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

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, threshold: int = 4) -> None:
        super().__init__(label)
        self.target = target
        self.threshold = threshold

    def _get_topple(self, height: Array, *around: Array, **kwargs: object) -> Array:
        """Return the heights after every site at or above the threshold topples once."""
        gained = np.sum([neighbor >= self.threshold for neighbor in around], axis=0)
        return height - self.threshold * (height >= self.threshold) + gained

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the toppling of the heights of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        neighborhood = RadiusNeighborhood(1, metric='manhattan', center=True)
        self.laws_ = [
            UpdateLaw(context.target.height, self._get_topple, neighborhood=neighborhood, label=self.label),
        ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import FixedEnergySandpileSystem
        >>> trajectory = FixedEnergySandpileSystem((8, 8), density=2.0).prepare().simulate(until=2)
        >>> trajectory.get_values('pile.height').shape
        (3, 64)
    """

    field = 'physics/criticality'
    topics = ('lattice', 'absorbing states', 'phase transition', 'conservation', 'sandpile')
    references = ('Dickman, Vespignani, and Zapperi 1998', 'Vespignani et al. 2000', 'Gros 2015')
    until = 500
    dt = 1
    observables = (Observable('activity', _get_active, group='pile'),)

    def __init__(
        self,
        shape: Sequence[int] = (32, 32),
        *,
        density: float = 2.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the pile, built from its model parameters.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        size = int(np.prod(self.shape))
        rng = np.random.default_rng(self.random_state)
        grains = rng.integers(size, size=round(self.density * size))
        heights = np.bincount(grains, minlength=size).astype(float)
        space = LatticeSpace(tuple(self.shape), boundary='periodic')
        self.system_ = System(
            self.label,
            elements=[GrainSiteElement('pile', height=heights, space=space)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.threshold = threshold
        self.random_state = random_state

    @staticmethod
    def _load(packet: float, energies: Array, *, random: np.random.Generator, **kwargs: object) -> Array:
        """Return the energies with a packet added on a random site.

        The packet has a uniform size in `[0, packet]`.
        """
        rng = random
        following = energies.copy()
        following[tuple(rng.integers(size) for size in energies.shape)] += rng.uniform(0.0, packet)
        return following

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Zhang sandpile, a continuous sandpile whose unstable sites give away all their energy.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
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
        self.system_ = System(
            self.label,
            elements=[EnergySiteElement('pile', energy=energies.ravel(), space=space)],
            interactions=[avalanche],
        )


class BranchingInteraction(Interaction):
    """Branching of an avalanche down a binary tree, with the branching probability tuned by the dissipation.

    Every step, each of the `z` active sites of generation `g` makes two active sites of the next generation with the
    probability `p`. The avalanche ends at the last generation or with no active site. Then the `z` sites of the last
    generation leave, `sigma` takes their number, `p` changes by `(1 - z) / N` with `N = 2^(generations + 1) - 1`
    sites, and a new avalanche starts from one site. Every avalanche is recorded as an `avalanche` event, whose profile
    holds the active sites of every generation before the last.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the tree, whose values are the probability `p`, the active sites `z`, the generation `g`, and the
            dissipation `sigma` of the last avalanche that ended.

        generations:
            The number of generations of the tree, at least 1.

    Examples:
        >>> from skcomplex.systems.physics.criticality import BranchingInteraction
        >>> BranchingInteraction('branching', target='tree').generations
        10
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, generations: int = 10) -> None:
        super().__init__(label)
        self.target = target
        self.generations = generations

    def _get_ends(self, tree: SimpleNamespace) -> Expression:
        """Return whether the avalanche of a tree ends this step.

        It ends at the last generation or with no active site.
        """
        return either(tree.g >= self.generations, tree.z == 0)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the active sites, the generation, and the branching probability of the next step.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        tree = context.target
        ends = self._get_ends(tree)
        sites = 2 ** (self.generations + 1) - 1
        dissipated = where(tree.g >= self.generations, tree.z, 0)
        self.laws_ = [
            UpdateLaw(
                tree.z,
                where(ends, 1, 2 * draw('children', 'binom', n=tree.z, p=tree.p)),
                activity=where(tree.g < self.generations, tree.z, 0),
                label='avalanche',
            ),
            UpdateLaw(tree.g, where(ends, 0, tree.g + 1)),
            UpdateLaw(tree.p, where(ends, tree.p + (1 - dissipated) / sites, tree.p)),
            UpdateLaw(tree.sigma, where(ends, dissipated, tree.sigma)),
        ]


class SelfOrganizedBranchingSystem(BaseSystem):
    """The self-organized branching process of Zapperi, Lauritsen, and Stanley.

    It is a mean-field sandpile.

    An avalanche starts from one active site at the top of a binary tree of `generations` generations. Every active site
    relaxes with the probability `p` into two active sites of the next generation, and stays still otherwise, so every
    generation has `2 Binomial(z, p)` active sites. The `sigma` active sites that reach the last generation leave the
    system as dissipation. After every avalanche one unit of energy has entered and `sigma` have left, so the
    probability changes by `(1 - sigma) / N` with `N = 2^(generations + 1) - 1` sites. A supercritical `p` makes large
    dissipating avalanches that lower it, and a subcritical one lets energy accumulate, so `p` organizes itself to the
    critical value `p_c = 1 / 2`, where the branching ratio `2 p` is 1. The process advances one generation per step, so
    a step is a time of the avalanche and avalanches follow each other without pause. Every avalanche is recorded as an
    `avalanche` event, whose profile holds the active sites of every generation before the last and whose size is
    their sum. The system has the probability `tree.p`, the active sites `tree.z`, the generation `tree.g`, and the
    dissipation `tree.sigma` of the last avalanche that ended.

    Args:
        p:
            The initial branching probability.

        generations:
            The number of generations of the tree, at least 1.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import SelfOrganizedBranchingSystem
        >>> SelfOrganizedBranchingSystem(0.4).prepare().system_.elements[0].values['p']
        0.4
        >>> trajectory = SelfOrganizedBranchingSystem(0.5).prepare().simulate(until=100, random_state=0)
        >>> trajectory.events[0].profile[0]
        1
    """

    field = 'physics/criticality'
    topics = ('branching process', 'avalanches', 'mean field', 'self-organized criticality', 'stochastic')
    references = ('Zapperi, Lauritsen, and Stanley 1995', 'Jensen 2023')
    until = 30000
    dt = 1

    def __init__(
        self,
        p: float = 0.3,
        *,
        generations: int = 10,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.p = p
        self.generations = generations

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the tree of the self-organized branching process.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('tree', values={'p': self.p, 'z': 1, 'g': 0, 'sigma': 0})],
            interactions=[BranchingInteraction('branching', target='tree', generations=self.generations)],
        )
