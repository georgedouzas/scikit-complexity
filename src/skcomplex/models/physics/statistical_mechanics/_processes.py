"""Lattice processes of statistical mechanics: Q2R, percolation, contact process, exclusion, and aggregation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import GroupElement, LatticeElement
from ....core.interactions import LocalRuleInteraction, TransferInteraction
from ....core.laws import Sink, Source
from ....core.laws.utils import bernoulli
from ....core.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from ....core.system import System
from ....types import Array
from ..._catalog import register_model
from ._contact_site import ContactSiteElement
from ._exclusion_site import ExclusionSiteElement
from ._percolation_site import BURNING, PercolationSiteElement
from ._q2r_spin import Q2RSpinElement
from ._walker import WalkerElement


def _flip_without_cost(*columns: Array) -> tuple[Array, Array]:
    """Return the spins and phases after one Q2R step, from the spins and the phases of a centred neighbourhood."""
    half = len(columns) // 2
    spin, around, phase = columns[0], columns[1:half], columns[half]
    flips = (phase == 1) & (sum(around) == 0)
    return np.where(flips, -spin, spin), 1 - phase


def _hop(s: SimpleNamespace) -> Expression:
    """Return 1 where a particle hops to the empty next site with probability `p`, one draw per edge, or 0."""
    return s.source.occupied * (1 - s.target.occupied) * bernoulli('hop', s.source.p)


def _enter(s: SimpleNamespace) -> Expression:
    """Return 1 where a particle enters the empty first site with probability `alpha`, and 0 otherwise."""
    return (1 - s.target.occupied) * bernoulli('entry', s.target.alpha)


def _leave(s: SimpleNamespace) -> Expression:
    """Return 1 where the particle on the last site leaves with probability `beta`, and 0 otherwise."""
    return s.source.occupied * bernoulli('exit', s.source.beta)


def _checkerboard(shape: tuple[int, ...]) -> Array:
    """Return 1 on the sites of even coordinate sum and 0 on the others."""
    return (np.indices(shape).sum(axis=0) % 2 == 0).astype(float)


@register_model(
    field='physics/statistical_mechanics',
    tags=('lattice', 'cellular automaton', 'reversible', 'spins'),
    references=('Vichniac 1984', 'Pomeau 1984', 'Lindgren 2024'),
    defaults={'until': 100, 'dt': 1},
)
def q2r(
    shape: Sequence[int] = (64, 64),
    *,
    up: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Q2R automaton, a reversible, energy-conserving Ising dynamics on a periodic lattice of even sides.

    The spins start uncorrelated with a fraction `up` of spins up, and the sites of even coordinate sum update first.

    Args:
        shape:
            The number of sites along each axis, all even.

        up:
            The probability that an initial spin is up.

        random_state:
            The seed or generator of the initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin` and phases `spins.phase`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import q2r
        >>> q2r((8,)).elements[0].check().prepare().size_
        8
    """
    shape = tuple(shape)
    spins = np.where(np.random.default_rng(random_state).random(shape) < up, 1.0, -1.0)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=Q2RSpinElement,
        space=space,
        values={'spin': spins, 'phase': _checkerboard(shape)},
    )
    flip = LocalRuleInteraction(
        'flip',
        target='spins',
        state=('spin', 'phase'),
        rule=_flip_without_cost,
        neighborhood=VonNeumannNeighborhood(n_dim=len(shape), center=True),
    )
    return System(elements=[group], interactions=[flip])


@register_model(
    field='physics/statistical_mechanics',
    tags=('lattice', 'percolation', 'phase transition', 'fire'),
    references=('Broadbent and Hammersley 1957', 'Stauffer and Aharony 1994', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def site_percolation(
    shape: Sequence[int] = (64, 64),
    *,
    p: float = 0.6,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return site percolation probed by a fire lit along the top row, which spans the lattice above `p_c = 0.5927`.

    Sites are occupied with probability `p`, and the fire burns the clusters connected to the top row, one layer of
    sites per step.

    Args:
        shape:
            The number of rows and columns.

        p:
            The occupation probability.

        random_state:
            The seed or generator of the occupations.

    Returns:
        system:
            The model, with the covers `lattice.cover`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import site_percolation
        >>> site_percolation((10, 10)).elements[0].check().prepare().size_
        100
    """
    shape = tuple(shape)
    cover = (np.random.default_rng(random_state).random(shape) < p).astype(float)
    cover[0] *= BURNING
    space = LatticeSpace(shape, boundary='fixed')
    return System(
        elements=[LatticeElement('lattice', element=PercolationSiteElement, space=space, values={'cover': cover})],
    )


@register_model(
    field='physics/statistical_mechanics',
    tags=('lattice', 'stochastic', 'absorbing state', 'phase transition', 'epidemics'),
    references=('Harris 1974', 'Marro and Dickman 1999', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 200, 'dt': 1},
)
def contact_process(
    shape: Sequence[int] = (64, 64),
    *,
    rate: float = 3.0,
    tau: float = 0.1,
    infected: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the contact process on a periodic lattice, which survives above a critical infection rate near 1.65.

    Below the critical rate the infection dies out and the lattice falls into the absorbing healthy state.

    Args:
        shape:
            The number of rows and columns.

        rate:
            The infection rate.

        tau:
            The time slice of one step.

        infected:
            The initial fraction of infected sites.

        random_state:
            The seed or generator of the initial infections.

    Returns:
        system:
            The model, with the infections `lattice.infected`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import contact_process
        >>> contact_process((10, 10), rate=1.0).elements[0].values['rate']
        1.0
    """
    shape = tuple(shape)
    states = (np.random.default_rng(random_state).random(shape) < infected).astype(float)
    space = LatticeSpace(shape)
    return System(
        elements=[
            LatticeElement(
                'lattice',
                element=ContactSiteElement,
                space=space,
                values={'infected': states, 'rate': rate, 'tau': tau},
            ),
        ],
    )


@register_model(
    field='physics/statistical_mechanics',
    tags=('lattice', 'stochastic', 'exclusion', 'traffic', 'nonequilibrium'),
    references=('Derrida, Domany, and Mukamel 1992', 'Schütz 2001', 'Helbing 2001'),
    defaults={'until': 500, 'dt': 1},
)
def tasep(
    n_sites: int = 200,
    *,
    p: float = 0.5,
    alpha: float = 0.3,
    beta: float = 0.7,
    ring: bool = False,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the totally asymmetric simple exclusion process with parallel update, on an open line or a ring.

    Every step all particles whose right neighbour is empty hop to it with probability `p` at once, a transfer along
    every edge of the lane with one draw per edge. On an open line particles enter the first site with probability
    `alpha` and leave the last site with probability `beta`. On a ring at density `rho` the current is
    `(1 - sqrt(1 - 4 p rho (1 - rho))) / 2`. On an open line the entry and exit probabilities select the low-density,
    high-density, or maximal-current phase.

    Args:
        n_sites:
            The number of sites.

        p:
            The probability of a hop.

        alpha:
            The probability of an entry at the left end of an open line.

        beta:
            The probability of an exit at the right end of an open line.

        ring:
            Whether the line is a ring instead of an open line.

        density:
            The initial fraction of occupied sites.

        random_state:
            The seed or generator of the initial occupations.

    Returns:
        system:
            The model, with the occupations `lane.occupied` and the hops `lane.hops`, and on an open line the
            entries `entry.flow_0` and the exits `exit.flow_0`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import tasep
        >>> [interaction.label for interaction in tasep(20).interactions]
        ['hops', 'entry', 'exit']
    """
    rng = np.random.default_rng(random_state)
    occupied = np.zeros(n_sites)
    occupied[rng.choice(n_sites, round(density * n_sites), replace=False)] = 1.0
    boundary = 'periodic' if ring else 'fixed'
    lane = LatticeElement(
        'lane',
        element=ExclusionSiteElement,
        space=LatticeSpace(n_sites, boundary=boundary),
        values={'occupied': occupied, 'p': p, 'alpha': alpha, 'beta': beta},
    )
    hops = TransferInteraction(
        'hops',
        source='lane',
        target='lane',
        state='occupied',
        amount=_hop,
        flow='hops',
        offset=1,
        boundary=boundary,
    )
    if ring:
        return System(elements=[lane], interactions=[hops], time=DiscreteTime())
    first, last = 'lane[0]', f'lane[{n_sites - 1}]'
    entry = TransferInteraction(
        'entry',
        source=first,
        target=first,
        state=Source(),
        into='occupied',
        amount=_enter,
    )
    leaving = TransferInteraction('exit', source=last, target=last, state='occupied', into=Sink(), amount=_leave)
    return System(elements=[lane], interactions=[hops, entry, leaving], time=DiscreteTime())


@register_model(
    field='physics/statistical_mechanics',
    tags=('agents', 'aggregation', 'fractal', 'random walk'),
    references=('Witten and Sander 1981', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def diffusion_limited_aggregation(
    n_walkers: int = 400,
    *,
    size: int = 41,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return diffusion-limited aggregation on a periodic grid, where random walkers stick to a seed at the center.

    Walkers that touch the aggregate join it, so the tips of the aggregate catch most walkers and it grows into a
    branched fractal of dimension about 1.7.

    Args:
        n_walkers:
            The number of walkers, the seed included.

        size:
            The side of the grid.

        random_state:
            The seed or generator of the initial cells of the walkers.

    Returns:
        system:
            The model, with the agent group `walkers`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import diffusion_limited_aggregation
        >>> diffusion_limited_aggregation(10, size=11).elements[0].check().prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    middle = size // 2
    cells = rng.integers(0, size, (n_walkers, 2))
    cells[0] = middle
    stuck = np.zeros(n_walkers)
    stuck[0] = 1.0
    walkers = GroupElement(
        'walkers',
        element=WalkerElement,
        size=n_walkers,
        values={'stuck': stuck},
        location=[tuple(map(int, cell)) for cell in cells],
    )
    return System(
        elements=[walkers],
        space=LatticeSpace((size, size), boundary='periodic', neighborhood='von_neumann'),
    )
