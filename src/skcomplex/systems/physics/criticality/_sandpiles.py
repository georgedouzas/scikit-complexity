"""Variants of self-organized criticality: the fixed-energy sandpile, the Zhang sandpile, and branching avalanches."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import CascadeInteraction, LocalRuleInteraction, MapInteraction
from ....core.components.laws.utils import draw, either, where
from ....core.components.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from ....core.components.system import System
from ....core.specs import ObservableSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ..common import SandpileCellElement


class BranchingTreeElement(Element):
    """Self-organized branching process: avalanches on a binary tree whose branching probability the avalanches tune.

    The probability `p` that an active site relaxes into two active sites of the next generation, the number `z` of
    active sites of the current generation of the avalanche, and its generation `g`. Map interactions advance the
    avalanche by one generation per step.

    Args:
        label:
            The label of the tree.

        visual:
            How the tree is drawn, or `None` for `default_visual_`.

        p:
            The initial branching probability.

        z:
            The initial number of active sites.

        g:
            The initial generation.

    Attributes:
        default_visual_:
            The visual of the tree when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import BranchingTreeElement
        >>> BranchingTreeElement('tree').p
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.3,
        z: Annotated[float, StateSpec(domain=NN)] = 1,
        g: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.z = z
        self.g = g


class EnergySiteElement(Element):
    """Site of the Zhang sandpile, which stores a continuous energy.

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
    """Parallel toppling of a sandpile without drive, where every site at or above the threshold topples once a step.

    Every step, every site that holds at least `threshold` grains loses `threshold` grains, and every site gains one
    grain from each toppling neighbour, all at once. On a periodic lattice no grain is lost, so the number of grains is
    conserved, and the activity either dies out in an absorbing configuration or lasts forever.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
            rule=self._topple,
            neighborhood=VonNeumannNeighborhood(center=True),
        )
        self.threshold = threshold

    def _topple(self, height: Array, *around: Array) -> Array:
        """Return the heights after every site at or above the threshold topples once."""
        gained = np.sum([neighbor >= self.threshold for neighbor in around], axis=0)
        return height - self.threshold * (height >= self.threshold) + gained


THRESHOLD = 4


def _active(site: SimpleNamespace) -> Expression:
    """Return 1 for a site that holds at least the threshold, an active site, and 0 otherwise."""
    return where(site.height >= THRESHOLD, 1, 0)


def _load(packet: float, energies: Array, *, random: np.random.Generator) -> Array:
    """Return the energies with a packet of uniform size in `[0, packet]` added on a random site."""
    following = energies.copy()
    following[tuple(random.integers(size) for size in energies.shape)] += random.uniform(0.0, packet)
    return following


def _ends(tree: SimpleNamespace, generations: int) -> Expression:
    """Return whether the avalanche of a tree ends this step, at the last generation or with no active site."""
    return either(tree.g >= generations, tree.z == 0)


def _branch(generations: int, s: SimpleNamespace) -> Expression:
    """Return the active sites of the next generation, or the one site of a new avalanche."""
    return where(_ends(s.target, generations), 1, 2 * draw('children', 'binom', n=s.target.z, p=s.target.p))


def _advance(generations: int, s: SimpleNamespace) -> Expression:
    """Return the next generation, or 0 for a new avalanche."""
    return where(_ends(s.target, generations), 0, s.target.g + 1)


def _tune(generations: int, s: SimpleNamespace) -> Expression:
    """Return the branching probability, changed by `(1 - sigma) / N` at the end of an avalanche."""
    sites = 2 ** (generations + 1) - 1
    dissipated = where(s.target.g >= generations, s.target.z, 0)
    return where(_ends(s.target, generations), s.target.p + (1 - dissipated) / sites, s.target.p)


@register_system(
    field='physics/criticality',
    tags=('lattice', 'absorbing states', 'phase transition', 'conservation', 'sandpile'),
    references=('Dickman, Vespignani, and Zapperi 1998', 'Vespignani et al. 2000', 'Gros 2015'),
    defaults={'until': 500, 'dt': 1},
    observables=lambda system: [ObservableSpec('activity', _active, group='pile')],
)
def fixed_energy_sandpile(
    shape: Sequence[int] = (32, 32),
    *,
    density: float = 2.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the fixed-energy sandpile, a Bak-Tang-Wiesenfeld pile with no drive and no loss on a periodic lattice.

    Every step, every site with at least four grains gives one grain to each of its four neighbours, all at once.
    The lattice is periodic, so the density of grains `density` is conserved, and it controls an absorbing phase
    transition. Below the critical density, about 2.125 on the square lattice, the topplings stop in an absorbing
    configuration, and above it the pile stays active forever. The density of active sites, the observable `activity`,
    is the order parameter. The grains start scattered uniformly at random over the sites.

    Args:
        shape:
            The number of rows and columns.

        density:
            The mean number of grains per site.

        random_state:
            The seed or generator of the initial grains.

    Returns:
        system:
            The system, with the heights `pile.height` and the observable `activity`, the density of active sites.

    Examples:
        >>> from skcomplex.systems.physics.criticality import fixed_energy_sandpile
        >>> fixed_energy_sandpile((8, 8), density=2.0).elements[0].prepare().size_
        64
    """
    size = int(np.prod(shape))
    rng = np.random.default_rng(random_state)
    grains = rng.integers(size, size=round(density * size))
    heights = np.bincount(grains, minlength=size).reshape(tuple(shape)).astype(float)
    space = LatticeSpace(tuple(shape), boundary='periodic')
    return System(
        elements=[LatticeElement('pile', element=SandpileCellElement, space=space, values={'height': heights})],
        interactions=[ToppleInteraction('topple', target='pile', threshold=THRESHOLD)],
    )


@register_system(
    field='physics/criticality',
    tags=('lattice', 'avalanches', 'power law', 'continuous energy', 'sandpile'),
    references=('Zhang 1989', 'Jensen 2023'),
    defaults={'until': 10000, 'dt': 100},
)
def zhang_sandpile(
    shape: Sequence[int] = (32, 32),
    *,
    threshold: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Zhang sandpile, a continuous sandpile whose unstable sites give away all their energy.

    Every step loads a site chosen at random with a packet of energy of uniform size in `[0, threshold]`. A site whose
    energy reaches the threshold relaxes: its energy falls to 0 and each of its four neighbours gains a quarter of it,
    all at once, until no site is unstable. Energy that would go beyond the open edges is lost. The pile organizes
    itself into a critical state with avalanches of every size, recorded as `avalanche` events with their sizes,
    durations, and areas, and the energies of the sites cluster near a few values. The energies start uniformly at
    random below the threshold.

    Args:
        shape:
            The number of rows and columns.

        threshold:
            The energy at which a site relaxes, which is also the largest packet.

        random_state:
            The seed or generator of the initial energies.

    Returns:
        system:
            The system, with the energies `pile.energy`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import zhang_sandpile
        >>> zhang_sandpile((8, 8)).interactions[0].mode
        'reset'
    """
    energies = np.random.default_rng(random_state).uniform(0.0, threshold, size=tuple(shape))
    space = LatticeSpace(tuple(shape), boundary='fixed')
    avalanche = CascadeInteraction(
        'avalanche',
        target='pile',
        state='energy',
        threshold=threshold,
        share=1 / 4,
        mode='reset',
        drive=partial(_load, threshold),
    )
    return System(
        elements=[LatticeElement('pile', element=EnergySiteElement, space=space, values={'energy': energies})],
        interactions=[avalanche],
    )


@register_system(
    field='physics/criticality',
    tags=('branching process', 'avalanches', 'mean field', 'self-organized criticality', 'stochastic'),
    references=('Zapperi, Lauritsen, and Stanley 1995', 'Jensen 2023'),
    defaults={'until': 30000, 'dt': 1},
)
def self_organized_branching(p: float = 0.3, *, generations: int = 10) -> System:
    """Return the self-organized branching process of Zapperi, Lauritsen, and Stanley, a mean-field sandpile.

    An avalanche starts from one active site at the top of a binary tree of `generations` generations. Every active
    site relaxes with the probability `p` into two active sites of the next generation, and stays still otherwise, so
    every generation has `2 Binomial(z, p)` active sites. The `sigma` active sites that reach the last generation leave
    the system as dissipation. After every avalanche one unit of energy has entered and `sigma` have left, so the
    probability changes by `(1 - sigma) / N` with `N = 2^(generations + 1) - 1` sites. A supercritical `p` makes large
    dissipating avalanches that lower it, and a subcritical one lets energy accumulate, so `p` organizes itself to the
    critical value `p_c = 1 / 2`, where the branching ratio `2 p` is 1. The process advances one generation per step,
    so a step is a time of the avalanche and avalanches follow each other without pause.

    Args:
        p:
            The initial branching probability.

        generations:
            The number of generations of the tree, at least 1.

    Returns:
        system:
            The system, with the probability `tree.p`, the active sites `tree.z`, and the generation `tree.g`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import self_organized_branching
        >>> self_organized_branching(0.4).elements[0].p
        0.4
    """
    return System(
        elements=[BranchingTreeElement('tree', p=p)],
        interactions=[
            MapInteraction('branching', target='tree', state='z', function=partial(_branch, generations)),
            MapInteraction('generation', target='tree', state='g', function=partial(_advance, generations)),
            MapInteraction('tuning', target='tree', state='p', function=partial(_tune, generations)),
        ],
        time=DiscreteTime(),
    )
