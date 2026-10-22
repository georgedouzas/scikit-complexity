"""Driven lattice particles: zero-range processes, the energy cascade on a tree, and exclusion with open boundaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, VisualSpec, parameter, role, state
from ....core.model import (
    ContinuousSpace,
    Element,
    Interaction,
    Jump,
    LatticeSpace,
    Law,
    Symbols,
    System,
    shift_member,
    where,
)
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]

TREE_ASPECT = 0.7


def _departure_rate(n: Expression, b: Expression) -> Expression:
    """Return the zero-range rate `u(n) = 1 + b / n` of a site with `n` particles, or 0 for an empty site."""
    return where(n > 0, 1 + b / where(n > 0, n, 1), 0)


def _scatter(n_particles: int, n_sites: int, random_state: int | np.random.Generator | None) -> Array:
    """Return the occupations of sites that received a number of particles one at a time on random sites."""
    sites = np.random.default_rng(random_state).integers(0, n_sites, n_particles)
    return np.bincount(sites, minlength=n_sites).astype(float)


def _level_start(level: int, branches: int) -> int:
    """Return the index of the first node of a level of a tree, counted from level 1, in breadth-first order."""
    return (branches ** (level - 1) - 1) // (branches - 1)


def _tree_layout(n_levels: int, branches: int) -> Array:
    """Return the drawing positions of the nodes of a tree in breadth-first order, on ellipses around the root."""
    rings = []
    for level in range(1, n_levels + 1):
        width = branches ** (level - 1)
        radius = (level - 1) / max(n_levels - 1, 1)
        angles = 2 * np.pi * (np.arange(width) + 0.5) / width
        rings.append(radius * np.column_stack([np.cos(angles), TREE_ASPECT * np.sin(angles)]))
    return np.concatenate(rings)


class ParticleSite(Element):
    """Site of a lattice of particles, holding the number `n` of particles or quanta on it.

    Args:
        label:
            The label of the site or the group.

        n:
            The initial number of particles.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ParticleSite
        >>> ParticleSite('site', n=3.0).n
        3.0
    """

    n: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)


class TreeNode(ParticleSite):
    """Node of a tree that holds a number `n` of energy quanta, drawn at a fixed position.

    Args:
        label:
            The label of the node or the group.

        n:
            The initial number of quanta.

        x:
            The position of the node in the drawing of the tree.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import TreeNode
        >>> TreeNode('root').n
        0.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec(
        'points',
        {'position': 'x', 'radius': 0.04, 'color_by': 'n', 'color_range': (0.0, 2.0), 'states': (), 'trail': 0},
    )
    x: npt.ArrayLike = state(default=0.0, vector=True)


class ZeroRangeHop(Interaction):
    """Hop of one particle along a link, forward at rate `p u(n)` and backward at rate `q u(n')`.

    The rate depends only on the occupation of the departure site, through `u(n) = 1 + b / n`.

    Args:
        label:
            The label of the link.

        link:
            The departure and the arrival site of a forward hop.

        p:
            The forward rate scale.

        q:
            The backward rate scale.

        b:
            The strength of the decrease of `u(n)` with the occupation, 0 for a constant rate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZeroRangeHop
        >>> ZeroRangeHop('hop', link=['sites[0]', 'sites[1]']).b
        4.0
    """

    link: list[str] = role(variadic=True)
    p: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    q: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=4.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the forward and the backward hop.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first, second = (site.n for site in s.link)
        return [
            Jump(s.p * _departure_rate(first, s.b), {first: -1, second: 1}),
            Jump(s.q * _departure_rate(second, s.b), {second: -1, first: 1}),
        ]


class ZeroRangeHops(Interaction):
    """Hops of one particle from every site of a ring, to the right at rate `p u(n)` and to the left at rate `q u(n)`.

    The rate depends only on the occupation `n` of the departure site, through `u(n) = 1 + b / n`.

    Args:
        label:
            The label of the hops.

        sites:
            The group of sites, in ring order.

        p:
            The rate scale of hops to the right.

        q:
            The rate scale of hops to the left.

        b:
            The strength of the decrease of `u(n)` with the occupation, 0 for a constant rate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZeroRangeHops
        >>> ZeroRangeHops('hops', sites='sites').b
        4.0
    """

    sites: str = role()
    p: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    q: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=4.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the hops to the right and to the left.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.sites.n
        rate = _departure_rate(n, s.b)
        return [
            Jump(s.p * rate, {n: -1, shift_member(n, 1): 1}),
            Jump(s.q * rate, {n: -1, shift_member(n, -1): 1}),
        ]


class ExclusionHops(Interaction):
    """Hops of a particle from every site of a line to an empty neighbour, to the right at rate `p` and left at `q`.

    No particle hops past the ends of the line.

    Args:
        label:
            The label of the hops.

        sites:
            The group of sites, in line order.

        p:
            The rate of hops to the right.

        q:
            The rate of hops to the left.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ExclusionHops
        >>> ExclusionHops('hops', sites='lane').p
        1.0
    """

    sites: str = role()
    p: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    q: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the hops to the right and to the left.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.sites.n
        right, left = (shift_member(n, step, boundary='fixed', fill=1.0) for step in (1, -1))
        return [
            Jump(s.p * n * (1 - right), {n: -1, shift_member(n, 1, boundary='fixed'): 1}),
            Jump(s.q * n * (1 - left), {n: -1, shift_member(n, -1, boundary='fixed'): 1}),
        ]


class Reservoir(Interaction):
    """Reservoir that adds one particle to each of its sites at rate `entry` and removes one at rate `exit`.

    A particle leaves only an occupied site. With exclusion a particle enters only an empty site.

    Args:
        label:
            The label of the reservoir.

        sites:
            The sites in contact with the reservoir.

        entry:
            The rate at which a particle enters a site.

        exit:
            The rate at which a particle leaves an occupied site.

        exclusion:
            Whether a site holds at most one particle.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import Reservoir
        >>> Reservoir('bath', sites=['lane[0]'], entry=0.5).exit
        0.0
    """

    sites: list[str] = role(variadic=True)
    entry: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    exit: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    exclusion: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the entry and the exit of a particle at every site.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Law] = []
        for site in s.sites:
            room = 1 - site.n if self.exclusion else 1
            laws += [Jump(s.entry * room, {site.n: 1}), Jump(s.exit * where(site.n > 0, 1, 0), {site.n: -1})]
        return laws


@register_model(
    field='physics/statistical_mechanics',
    tags=('zero-range process', 'condensation', 'nonequilibrium', 'jump process'),
    references=('Evans and Hanney 2005', 'Bertin 2021'),
    defaults={'until': 2000.0, 'dt': 10.0, 'backend': 'numba'},
)
def zero_range_process(
    n_sites: int = 50,
    *,
    density: float = 3.0,
    b: float = 4.0,
    p: float = 1.0,
    q: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the zero-range process on a ring, which condenses above the critical density `1 / (b - 2)` for `b > 2`.

    A particle leaves a site with `n` particles at rate `u(n) = 1 + b / n`, to the right with weight `p` and to the
    left with weight `q`. The stationary distribution factorizes with the weights `f(n) = prod_k 1 / u(k)`, which
    decay as `n^-b`. Above the critical density a fluid at density `1 / (b - 2)` coexists with one site that holds the
    excess particles.

    Args:
        n_sites:
            The number of sites.

        density:
            The number of particles per site.

        b:
            The strength of the decrease of the departure rate with the occupation.

        p:
            The rate scale of hops to the right.

        q:
            The rate scale of hops to the left.

        random_state:
            The seed or generator of the initial sites of the particles.

    Returns:
        system:
            The model, with the occupations `sites.n`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import zero_range_process
        >>> system = zero_range_process(10, density=2.0)
        >>> float(system.elements[0].values['n'].sum()), len(system.interactions)
        (20.0, 1)
    """
    occupations = _scatter(round(density * n_sites), n_sites, random_state)
    sites = ParticleSite.lattice('sites', LatticeSpace(n_sites), n=occupations)
    return System(components=[sites], interactions=[ZeroRangeHops('hops', sites='sites', p=p, q=q, b=b)])


@register_model(
    field='physics/statistical_mechanics',
    tags=('zero-range process', 'energy cascade', 'nonequilibrium', 'tree', 'jump process'),
    references=('Bertin 2021', 'Bertin 2005'),
    defaults={'until': 200.0, 'dt': 1.0, 'backend': 'numba'},
)
def dissipative_zero_range(
    n_levels: int = 5,
    *,
    branches: int = 2,
    nu: float = 1.0,
    alpha: float = 0.5,
    beta_ext: float = 1.0,
    epsilon0: float = 1.0,
    dissipation: float = 1.0,
) -> System:
    """Return the dissipative zero-range process on a tree, with energy injected at the root and lost at the leaves.

    Level `j` of the tree has `m^(j - 1)` nodes, each with `m` branches to the next level. A quantum `epsilon0` of
    energy crosses a branch between levels `j` and `j + 1` at rate `nu_j = nu k_j^alpha`, `k_j = m^(j - 1)`, in either
    direction from an occupied node. A reservoir at the inverse temperature `beta_ext` exchanges quanta with the root,
    adding one at rate `nu exp(-beta_ext epsilon0)` and removing one at rate `nu`, and every leaf loses a quantum at the
    rate `dissipation`. The stationary distribution is a product of geometric laws `z_j^n`, with `z_j` the solution of
    the linear balance of the currents `nu_j (z_j - z_(j + 1))` at every level, and the injected flux is
    `nu (exp(-beta_ext epsilon0) - z_1)`.

    Args:
        n_levels:
            The number of levels `M`.

        branches:
            The number `m` of branches of every node above the leaves.

        nu:
            The rate scale of transfers and of the exchange with the reservoir.

        alpha:
            The exponent of the growth of the transfer rate with the pseudo wave number `k_j`.

        beta_ext:
            The inverse temperature of the reservoir.

        epsilon0:
            The quantum of energy.

        dissipation:
            The rate at which every leaf loses a quantum.

    Returns:
        system:
            The model, with the quanta `nodes.n` of the nodes in breadth-first order, from the root to the leaves.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import dissipative_zero_range
        >>> system = dissipative_zero_range(3, branches=3)
        >>> system.elements[0].size, len(system.interactions)
        (13, 14)
    """
    n_nodes = _level_start(n_levels + 1, branches)
    nodes = TreeNode.group('nodes', n_nodes, x=_tree_layout(n_levels, branches))
    interactions: list[Interaction] = []
    for level in range(1, n_levels):
        rate = nu * float(branches ** (level - 1)) ** alpha
        start, below = _level_start(level, branches), _level_start(level + 1, branches)
        for index in range(branches ** (level - 1)):
            for branch in range(branches):
                child = below + branches * index + branch
                link = [f'nodes[{start + index}]', f'nodes[{child}]']
                interactions.append(ZeroRangeHop(f'transfer_{start + index}_{child}', link=link, p=rate, q=rate, b=0.0))
    leaves = [f'nodes[{index}]' for index in range(_level_start(n_levels, branches), n_nodes)]
    interactions += [
        Reservoir('injection', sites=['nodes[0]'], entry=nu * np.exp(-beta_ext * epsilon0), exit=nu),
        Reservoir('dissipation', sites=leaves, exit=dissipation),
    ]
    return System(components=[nodes], interactions=interactions, space=ContinuousSpace(n_dim=2))


@register_model(
    field='physics/statistical_mechanics',
    tags=('exclusion', 'nonequilibrium', 'phase transition', 'traffic', 'jump process'),
    references=('Derrida, Evans, Hakim, and Pasquier 1993', 'Bertin 2021'),
    defaults={'until': 500.0, 'dt': 2.0, 'backend': 'numba'},
)
def asep(
    n_sites: int = 100,
    *,
    alpha: float = 0.3,
    gamma: float = 0.7,
    p: float = 1.0,
    q: float = 0.0,
    density: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the asymmetric simple exclusion process on an open line, in continuous time.

    A particle hops to the empty right neighbour at rate `p` and to the empty left neighbour at rate `q`, enters the
    empty first site at rate `alpha`, and leaves the last site at rate `gamma`, the exit rate that Bertin writes
    `gamma` and others write `beta`. For `p = 1` and `q = 0` the current is `alpha (1 - alpha)` in the low-density
    phase `alpha < 1/2, gamma > alpha`, `gamma (1 - gamma)` in the high-density phase `gamma < 1/2, gamma < alpha`, and
    `1/4` in the maximal-current phase `alpha, gamma > 1/2`.

    Args:
        n_sites:
            The number of sites.

        alpha:
            The entry rate.

        gamma:
            The exit rate.

        p:
            The rate of hops to the right.

        q:
            The rate of hops to the left.

        density:
            The initial fraction of occupied sites.

        random_state:
            The seed or generator of the initial occupations.

    Returns:
        system:
            The model, with the occupations `lane.n`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import asep
        >>> len(asep(10).interactions)
        3
    """
    occupied = (np.random.default_rng(random_state).random(n_sites) < density).astype(float)
    lane = ParticleSite.lattice('lane', LatticeSpace(n_sites, boundary='fixed'), n=occupied)
    interactions = [
        ExclusionHops('hops', sites='lane', p=p, q=q),
        Reservoir('entry', sites=['lane[0]'], entry=alpha, exclusion=True),
        Reservoir('exit', sites=[f'lane[{n_sites - 1}]'], exit=gamma, exclusion=True),
    ]
    return System(components=[lane], interactions=interactions)
