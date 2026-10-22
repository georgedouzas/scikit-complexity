"""Lattice processes of statistical mechanics: Q2R, percolation, contact process, exclusion, and aggregation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Drive, Law, LocalRule, Rule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import System

type Array = npt.NDArray[np.float64]

BURNING = 2
BURNT = 3


def _flip_without_cost(*columns: Array) -> tuple[Array, Array]:
    """Return the spins and phases after one Q2R step, from the spins and the phases of a centred neighbourhood."""
    half = len(columns) // 2
    spin, around, phase = columns[0], columns[1:half], columns[half]
    flips = (phase == 1) & (sum(around) == 0)
    return np.where(flips, -spin, spin), 1 - phase


def _burn(cover: Array, *around: Array) -> Array:
    """Return the next cover of a site, which burns out, catches fire from a burning neighbour, or stays."""
    catches = (cover == 1) & np.any(np.stack(around) == BURNING, axis=0)
    return np.where(cover == BURNING, BURNT, np.where(catches, BURNING, cover))


def _spread(infected: Array, *around: Array, rate: Any, tau: Any, random: Array) -> Array:  # noqa: ANN401
    """Return the next infections of the contact process over a time slice `tau`."""
    share = sum(around) / len(around)
    infects = random < -np.expm1(-rate * tau * share)
    recovers = random < -np.expm1(-tau)
    return np.where(infected == 1, np.where(recovers, 0.0, 1.0), np.where(infects, 1.0, 0.0))


def _hop_ring(occupied: Array, *, random: np.random.Generator, p: Any) -> Array:  # noqa: ANN401
    """Return the occupations of a ring after one parallel update of the exclusion process."""
    hops = (
        (occupied == 1) & (np.roll(occupied, -1) == 0) & (random.random(occupied.shape) < p)  # nosec B311: seeded NumPy
    )
    return occupied - hops + np.roll(hops, 1)


def _hop_open(occupied: Array, *, random: np.random.Generator, p: Any, alpha: Any, beta: Any) -> Array:  # noqa: ANN401
    """Return the occupations of an open line after one parallel update with entry at the left and exit at the right."""
    free = np.append(occupied[1:], 1.0) == 0
    hops = (occupied == 1) & free & (random.random(occupied.shape) < p)  # nosec B311: seeded NumPy generator
    exits = (occupied[-1] == 1) & (random.random() < np.ravel(beta)[-1])  # nosec B311: seeded NumPy generator
    enters = (occupied[0] == 0) & (random.random() < np.ravel(alpha)[0])  # nosec B311: seeded NumPy generator
    following = occupied - hops
    following[1:] += hops[:-1]
    following[-1] -= exits
    following[0] += enters
    return following


class Q2RSpin(Element):
    """Site of the Q2R automaton, a spin and a phase that says whether the site updates in the current step.

    Every step the updating sites flip their spin when their neighbours sum to zero, which leaves the Ising energy
    unchanged, and every site swaps its phase, so the two sublattices update in turn and the dynamics is reversible.

    Args:
        label:
            The label of the site or the lattice group.

        spin:
            The initial spin, 1 or -1.

        phase:
            The initial phase, 1 for updating and 0 for quiescent.

        n_dim:
            The number of dimensions of the lattice.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import Q2RSpin
        >>> Q2RSpin('site').n_dim
        2
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    phase: float = state(default=0.0, domain=Domain.BOOLEAN)
    n_dim: int = 2

    def laws(self, s: Symbols) -> list[Law]:
        """Return the reversible energy-conserving flip and the swap of phases.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        nearest = Neighborhood.von_neumann(n_dim=self.n_dim, center=True)
        return [LocalRule((s.spin, s.phase), nearest, _flip_without_cost, reads=(s.spin, s.phase))]


class PercolationSite(Element):
    """Site of a percolation lattice: empty at 0, occupied at 1, burning at 2, or burnt at 3.

    Fire spreads from burning sites to their occupied neighbours and burns out in one step, so a fire lit along one
    edge burns exactly the clusters that touch that edge.

    Args:
        label:
            The label of the site or the lattice group.

        cover:
            The initial cover, 0, 1, 2, or 3.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import PercolationSite
        >>> PercolationSite('site').cover
        0.0
    """

    cover: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the spread and burning out of the fire.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.cover, Neighborhood.von_neumann(center=True), _burn)]


class ContactSite(Element):
    """Site of the contact process, which infects a neighbour at the rate `rate / 4` and recovers at rate 1.

    One step is a time slice `tau`, in which an infected site recovers with probability `1 - exp(-tau)` and a
    healthy site is infected with probability `1 - exp(-rate tau f)`, where `f` is the fraction of its infected
    neighbours.

    Args:
        label:
            The label of the site or the lattice group.

        infected:
            The initial state, 1 for infected and 0 for healthy.

        rate:
            The infection rate.

        tau:
            The time slice of one step.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ContactSite
        >>> ContactSite('site').rate
        3.0
    """

    infected: float = state(default=0.0, domain=Domain.BOOLEAN)
    rate: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=0.1, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stochastic infection and recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.infected, Neighborhood.von_neumann(center=True), _spread, stochastic=True)]


class ExclusionSite(Element):
    """Site of the totally asymmetric simple exclusion process, empty at 0 or holding one particle at 1.

    Every step all particles whose right neighbour is empty hop to it with probability `p` at once. On an open
    line particles enter the first site with probability `alpha` and leave the last site with probability `beta`.

    Args:
        label:
            The label of the site or the lattice group.

        occupied:
            The initial occupation.

        p:
            The probability of a hop.

        alpha:
            The probability of an entry at the left end of an open line.

        beta:
            The probability of an exit at the right end of an open line.

        ring:
            Whether the line is a ring instead of an open line.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ExclusionSite
        >>> ExclusionSite('site').ring
        False
    """

    occupied: float = state(default=0.0, domain=Domain.BOOLEAN)
    p: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.7, domain=Domain.NONNEGATIVE)
    ring: bool = False

    def laws(self, s: Symbols) -> list[Law]:
        """Return the parallel update of hops, entries, and exits.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Drive(s.occupied, _hop_ring if self.ring else _hop_open)]


class Walker(Element):
    """Random walker of diffusion-limited aggregation, which sticks when it reaches a stuck walker.

    Args:
        label:
            The label of the agent or group.

        stuck:
            Whether the walker belongs to the aggregate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import Walker
        >>> Walker('walker').stuck
        0.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'stuck', 'color_range': (0.0, 1.0)})
    stuck: float = state(default=0.0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the walk-and-stick rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.walk)]

    def walk(self, agent: AgentView, model: ModelView) -> None:
        """Stick next to the aggregate, or step to a random neighbouring cell.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if agent.stuck:
            return
        if any(other.stuck for other in agent.neighbors()):
            agent.stuck = 1.0
            return
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])


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
        >>> q2r((8,)).elements[0].size
        8
    """
    shape = tuple(shape)
    spins = np.where(np.random.default_rng(random_state).random(shape) < up, 1.0, -1.0)
    space = LatticeSpace(shape)
    group = Q2RSpin.lattice('spins', space, spin=spins, phase=_checkerboard(shape), n_dim=len(shape))
    return System(components=[group])


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
        >>> site_percolation((10, 10)).elements[0].size
        100
    """
    shape = tuple(shape)
    cover = (np.random.default_rng(random_state).random(shape) < p).astype(float)
    cover[0] *= BURNING
    space = LatticeSpace(shape, boundary='fixed')
    return System(components=[PercolationSite.lattice('lattice', space, cover=cover)])


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
    return System(components=[ContactSite.lattice('lattice', space, infected=states, rate=rate, tau=tau)])


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

    On a ring at density `rho` the current is `(1 - sqrt(1 - 4 p rho (1 - rho))) / 2`. On an open line the entry
    and exit probabilities select the low-density, high-density, or maximal-current phase.

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
            The model, with the occupations `lane.occupied`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import tasep
        >>> tasep(20, ring=True).elements[0].values['ring']
        True
    """
    rng = np.random.default_rng(random_state)
    occupied = np.zeros(n_sites)
    occupied[rng.choice(n_sites, round(density * n_sites), replace=False)] = 1.0
    space = LatticeSpace(n_sites, boundary='periodic' if ring else 'fixed')
    lane = ExclusionSite.lattice('lane', space, occupied=occupied, p=p, alpha=alpha, beta=beta, ring=ring)
    return System(components=[lane])


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
        >>> diffusion_limited_aggregation(10, size=11).elements[0].size
        10
    """
    rng = np.random.default_rng(random_state)
    middle = size // 2
    cells = rng.integers(0, size, (n_walkers, 2))
    cells[0] = middle
    stuck = np.zeros(n_walkers)
    stuck[0] = 1.0
    walkers = Walker.group('walkers', n_walkers, location=[tuple(map(int, cell)) for cell in cells], stuck=stuck)
    return System(
        components=[walkers],
        space=LatticeSpace((size, size), boundary='periodic', neighborhood='von_neumann'),
    )
