"""Driven lattice particles: zero-range processes, the energy cascade on a tree, and exclusion with open boundaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement, LatticeElement
from ....core.components.interactions import BaseInteraction
from ....core.components.spaces import ContinuousSpace, LatticeSpace
from ....core.components.system import System
from ....core.engine.backends import NumbaBackend
from ....core.types import Array
from ..._catalog import register_system
from ._exclusion_hops import ExclusionHopsInteraction
from ._particle_site import ParticleSiteElement
from ._reservoir import ReservoirInteraction
from ._tree_node import TreeNodeElement
from ._zero_range_hop import ZeroRangeHopInteraction
from ._zero_range_hops import ZeroRangeHopsInteraction

TREE_ASPECT = 0.7


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


@register_system(
    field='physics/statistical_mechanics',
    tags=('zero-range process', 'condensation', 'nonequilibrium', 'jump process'),
    references=('Evans and Hanney 2005', 'Bertin 2021'),
    defaults={'until': 2000.0, 'dt': 10.0, 'backend': NumbaBackend()},
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
            The system, with the occupations `sites.n`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import zero_range_process
        >>> system = zero_range_process(10, density=2.0)
        >>> float(system.elements[0].values['n'].sum()), len(system.interactions)
        (20.0, 1)
    """
    occupations = _scatter(round(density * n_sites), n_sites, random_state)
    sites = LatticeElement('sites', element=ParticleSiteElement, space=LatticeSpace(n_sites), values={'n': occupations})
    return System(elements=[sites], interactions=[ZeroRangeHopsInteraction('hops', sites='sites', p=p, q=q, b=b)])


@register_system(
    field='physics/statistical_mechanics',
    tags=('zero-range process', 'energy cascade', 'nonequilibrium', 'tree', 'jump process'),
    references=('Bertin 2021', 'Bertin 2005'),
    defaults={'until': 200.0, 'dt': 1.0, 'backend': NumbaBackend()},
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
            The system, with the quanta `nodes.n` of the nodes in breadth-first order, from the root to the leaves.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import dissipative_zero_range
        >>> system = dissipative_zero_range(3, branches=3)
        >>> system.elements[0].size, len(system.interactions)
        (13, 14)
    """
    n_nodes = _level_start(n_levels + 1, branches)
    nodes = GroupElement('nodes', element=TreeNodeElement, size=n_nodes, values={'x': _tree_layout(n_levels, branches)})
    interactions: list[BaseInteraction] = []
    for level in range(1, n_levels):
        rate = nu * float(branches ** (level - 1)) ** alpha
        start, below = _level_start(level, branches), _level_start(level + 1, branches)
        for index in range(branches ** (level - 1)):
            for branch in range(branches):
                child = below + branches * index + branch
                link = [f'nodes[{start + index}]', f'nodes[{child}]']
                interactions.append(
                    ZeroRangeHopInteraction(f'transfer_{start + index}_{child}', link=link, p=rate, q=rate, b=0.0),
                )
    leaves = [f'nodes[{index}]' for index in range(_level_start(n_levels, branches), n_nodes)]
    interactions += [
        ReservoirInteraction('injection', sites=['nodes[0]'], inflow=nu * np.exp(-beta_ext * epsilon0), outflow=nu),
        ReservoirInteraction('dissipation', sites=leaves, outflow=dissipation),
    ]
    return System(elements=[nodes], interactions=interactions, space=ContinuousSpace(n_dim=2))


@register_system(
    field='physics/statistical_mechanics',
    tags=('exclusion', 'nonequilibrium', 'phase transition', 'traffic', 'jump process'),
    references=('Derrida, Evans, Hakim, and Pasquier 1993', 'Bertin 2021'),
    defaults={'until': 500.0, 'dt': 2.0, 'backend': NumbaBackend()},
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
            The system, with the occupations `lane.n`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import asep
        >>> len(asep(10).interactions)
        3
    """
    occupied = (np.random.default_rng(random_state).random(n_sites) < density).astype(float)
    lane = LatticeElement(
        'lane',
        element=ParticleSiteElement,
        space=LatticeSpace(n_sites, boundary='fixed'),
        values={'n': occupied},
    )
    interactions = [
        ExclusionHopsInteraction('hops', sites='lane', p=p, q=q),
        ReservoirInteraction('entry', sites=['lane[0]'], inflow=alpha, exclusion=True),
        ReservoirInteraction('exit', sites=[f'lane[{n_sites - 1}]'], outflow=gamma, exclusion=True),
    ]
    return System(elements=[lane], interactions=interactions)
