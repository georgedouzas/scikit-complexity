"""Lattice processes of statistical mechanics.

The module covers Q2R, percolation, contact process, exclusion, and aggregation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction, TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw, Sink, Source, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.spaces import BaseSpace, DiscreteTime, LatticeSpace, RadiusNeighborhood
from ....core.components.systems import BaseSystem as OldBaseSystem
from ....core.components.systems import CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.specs import ParameterSpec, Role, StateSpec, VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array

BURNING = 2
BURNT = 3
NEAREST = RadiusNeighborhood(1, metric='manhattan', center=True)


class ContactSiteElement(Element):
    """Sites of the contact process.

    Every site is infected at 1 or healthy at 0, and holds the infection rate `rate` and the time slice `tau` of one
    step, which `ContactInfectionInteraction` reads.

    Args:
        label:
            The label of the site or the lattice of sites.

        infected:
            The initial state, 1 for infected and 0 for healthy, or one state per site.

        rate:
            The infection rate.

        tau:
            The time slice of one step.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ContactSiteElement
        >>> ContactSiteElement('site').rate
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        infected: object = 0.0,
        rate: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 0.1,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.infected = infected
        self.rate = rate
        self.tau = tau


class ContactInfectionInteraction(Interaction):
    """Infection and recovery of the contact process.

    A site infects a neighbour at the rate `rate / 4` and recovers at rate 1. One step is a time slice `tau`, in which
    an infected site recovers with probability `1 - exp(-tau)` and a healthy site is infected with probability
    `1 - exp(-rate tau f)`, where `f` is the fraction of its infected neighbours. It is an epidemic on a grid of
    villages, the spread of a plant across a field, or a rumour among neighbours.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the sites, whose values are `infected`, `rate`, and `tau`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ContactInfectionInteraction
        >>> ContactInfectionInteraction('infection', target='lattice').target
        'lattice'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _spread(infected: Array, *around: Array, rate: float | Array, tau: float | Array, random: Array) -> Array:
        """Return the next infections of the contact process over a time slice `tau`."""
        share = sum(around) / len(around)
        infects = random < -np.expm1(-rate * tau * share)
        recovers = random < -np.expm1(-tau)
        return np.where(infected == 1, np.where(recovers, 0.0, 1.0), np.where(infects, 1.0, 0.0))

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the stochastic infection and recovery of the sites.

        Args:
            context:
                The namespace of the target and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target.infected, self._spread, neighborhood=NEAREST)]


class ExclusionSiteElement(Element):
    """Site of the totally asymmetric simple exclusion process.

    The site is empty at 0 or holds one particle at 1.

    It holds the probabilities of a hop, of an entry at the left end of an open line, and of an exit at its right end,
    which the transfers of the process read.

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

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ExclusionSiteElement
        >>> ExclusionSiteElement('site').p
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        occupied: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
    ) -> None:
        super().__init__(label)
        self.occupied = occupied
        self.p = p
        self.alpha = alpha
        self.beta = beta


class PercolationSiteElement(Element):
    """Sites of a percolation lattice.

    Every site is empty at 0, occupied at 1, burning at 2, or burnt at 3.

    Args:
        label:
            The label of the site or the lattice of sites.

        cover:
            The initial cover, 0, 1, 2, or 3, or one cover per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import PercolationSiteElement
        >>> PercolationSiteElement('site').cover
        0.0
    """

    def __init__(self, label: str, *, cover: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.cover = cover


class FireSpreadInteraction(Interaction):
    """Fire on a percolation lattice.

    Fire spreads from burning sites to their occupied neighbours and burns out in one step, so a fire lit along one
    edge burns exactly the clusters that touch that edge.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the sites, whose value `cover` is 0, 1, 2, or 3.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import FireSpreadInteraction
        >>> FireSpreadInteraction('fire', target='lattice').target
        'lattice'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _burn(cover: Array, *around: Array) -> Array:
        """Return the next cover of a site.

        The site burns out, catches fire from a burning neighbour, or stays.
        """
        catches = (cover == 1) & np.any(np.stack(around) == BURNING, axis=0)
        return np.where(cover == BURNING, BURNT, np.where(catches, BURNING, cover))

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the spread and the burning out of the fire.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target.cover, self._burn, neighborhood=NEAREST, label=self.label)]


class Q2RSpinElement(Element):
    """Sites of the Q2R automaton.

    Every site holds a spin and a phase that says whether the site updates in the current step.

    Args:
        label:
            The label of the site or the lattice of sites.

        spin:
            The initial spin, 1 or -1, or one spin per site.

        phase:
            The initial phase, 1 for updating and 0 for quiescent, or one phase per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import Q2RSpinElement
        >>> Q2RSpinElement('site').phase
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        spin: object = 1.0,
        phase: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.spin = spin
        self.phase = phase


class Q2RFlipInteraction(Interaction):
    """Flips of the Q2R automaton.

    The local rule of Q2R flips the spins of the updating sites whose neighbours sum to zero, which leaves the Ising
    energy unchanged, and swaps the phase of every site, so the two sublattices update in turn and the dynamics is
    reversible.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the sites, whose values are the spins `spin` and the phases `phase`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import Q2RFlipInteraction
        >>> Q2RFlipInteraction('flip', target='spins').target
        'spins'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _flip_without_cost(*columns: Array) -> tuple[Array, Array]:
        """Return the spins and phases after one Q2R step.

        The step reads the spins and the phases of a centred neighbourhood.
        """
        half = len(columns) // 2
        spin, around, phase = columns[0], columns[1:half], columns[half]
        flips = (phase == 1) & (sum(around) == 0)
        return np.where(flips, -spin, spin), 1 - phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the flips of the spins and the swap of the phases.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            sites = context.target
            spin_and_phase = (sites.spin, sites.phase)
            self.laws_ = [UpdateLaw(spin_and_phase, self._flip_without_cost, neighborhood=NEAREST, label=self.label)]


class WalkerView(AgentView):
    """Walker of diffusion-limited aggregation as rules see it.

    The walker holds whether it belongs to the aggregate.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    stuck: float


class WalkerElement(Element):
    """Random walker of diffusion-limited aggregation.

    The walker sticks when it reaches a stuck walker.

    Args:
        label:
            The label of the agent or group.

        stuck:
            Whether the walker belongs to the aggregate.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import WalkerElement
        >>> WalkerElement('walker').stuck
        0.0
    """

    visual = VisualSpec('agents', {'color_by': 'stuck', 'color_range': (0.0, 1.0)})

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        stuck: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.stuck = stuck

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the walk-and-stick rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.walk)]

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return WalkerView

    def walk(self, agent: WalkerView, system: SystemView) -> None:
        """Stick next to the aggregate, or step to a random neighbouring cell.

        Args:
            agent:
                The walker.

            system:
                The system.
        """
        if agent.stuck:
            return
        if any(isinstance(other, WalkerView) and other.stuck for other in agent.get_neighbors()):
            agent.stuck = 1.0
            return
        cells = agent.get_neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])


class Q2rSystem(BaseSystem):
    """The Q2R automaton.

    It is a reversible, energy-conserving Ising dynamics on a periodic lattice of even sides.

    The spins start uncorrelated with a fraction `up` of spins up, and the sites of even coordinate sum update first.
    The system has the spins `spins.spin` and phases `spins.phase`.

    Args:
        shape:
            The number of sites along each axis, all even.

        up:
            The probability that an initial spin is up.

        random_state:
            The seed or generator of the initial spins.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import Q2rSystem
        >>> trajectory = Q2rSystem((8,)).prepare().simulate(until=1)
        >>> trajectory.get_values('spins.spin').shape
        (2, 8)
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'cellular automaton', 'reversible', 'spins')
    references = ('Vichniac 1984', 'Pomeau 1984', 'Lindgren 2024')
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        up: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.up = up
        self.random_state = random_state

    @staticmethod
    def _get_checkerboard(shape: tuple[int, ...]) -> Array:
        """Return 1 on the sites of even coordinate sum and 0 on the others."""
        return (np.indices(shape).sum(axis=0) % 2 == 0).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the spins and phases of the lattice and their flips.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        spins = np.where(np.random.default_rng(self.random_state).random(shape) < self.up, 1.0, -1.0)
        sites = Q2RSpinElement(
            'spins',
            spin=spins.ravel(),
            phase=self._get_checkerboard(shape).ravel(),
            space=LatticeSpace(shape),
        )
        self.system_ = System(self.label, elements=[sites], interactions=[Q2RFlipInteraction('flip', target='spins')])


class SitePercolationSystem(BaseSystem):
    """Site percolation probed by a fire lit along the top row.

    The fire spans the lattice above `p_c = 0.5927`.

    Sites are occupied with probability `p`, and the fire burns the clusters connected to the top row, one layer of
    sites per step. The system has the covers `lattice.cover`.

    Args:
        shape:
            The number of rows and columns.

        p:
            The occupation probability.

        random_state:
            The seed or generator of the occupations.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import SitePercolationSystem
        >>> trajectory = SitePercolationSystem((10, 10)).prepare().simulate(until=1)
        >>> trajectory.get_values('lattice.cover').shape
        (2, 100)
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'percolation', 'phase transition', 'fire')
    references = ('Broadbent and Hammersley 1957', 'Stauffer and Aharony 1994', 'Charbonneau 2017')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        p: float = 0.6,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.p = p
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the occupied sites with the fire lit along the top row.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        cover = (np.random.default_rng(self.random_state).random(shape) < self.p).astype(float)
        cover[0] *= BURNING
        sites = PercolationSiteElement('lattice', cover=cover.ravel(), space=LatticeSpace(shape, boundary='fixed'))
        fire = FireSpreadInteraction('fire', target='lattice')
        self.system_ = System(self.label, elements=[sites], interactions=[fire])


class ContactProcessSystem(BaseSystem):
    """The contact process on a periodic lattice.

    The process survives above a critical infection rate near 1.65.

    Below the critical rate the infection dies out and the lattice falls into the absorbing healthy state. The system
    has the infections `lattice.infected`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ContactProcessSystem
        >>> ContactProcessSystem((10, 10), rate=1.0).prepare().system_.elements[0].rate
        1.0
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'stochastic', 'absorbing state', 'phase transition', 'epidemics')
    references = ('Harris 1974', 'Marro and Dickman 1999', 'Pastor-Satorras et al. 2015')
    until = 200
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        rate: float = 3.0,
        tau: float = 0.1,
        infected: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.rate = rate
        self.tau = tau
        self.infected = infected
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the contact process on a lattice, with random infected sites at the start.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        states = (np.random.default_rng(self.random_state).random(shape) < self.infected).astype(float)
        lattice = ContactSiteElement(
            'lattice',
            infected=states.ravel(),
            rate=self.rate,
            tau=self.tau,
            space=LatticeSpace(shape),
        )
        infection = ContactInfectionInteraction('infection', target='lattice')
        self.system_ = System(self.label, elements=[lattice], interactions=[infection])


class TASEPSystem(OldBaseSystem):
    """The totally asymmetric simple exclusion process with parallel update.

    The process runs on an open line or a ring.

    Every step all particles whose right neighbour is empty hop to it with probability `p` at once, a transfer along
    every edge of the lane with one draw per edge. On an open line particles enter the first site with probability
    `alpha` and leave the last site with probability `beta`. On a ring at density `rho` the current is `(1 - sqrt(1 - 4
    p rho (1 - rho))) / 2`. On an open line the entry and exit probabilities select the low-density, high-density, or
    maximal-current phase. The system has the occupations `lane.occupied` and the hops `lane.hops`, and on an open line
    the entries `entry.flow_0` and the exits `exit.flow_0`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TASEPSystem
        >>> [interaction.label for interaction in TASEPSystem(20).prepare().system_.interactions]
        ['hops', 'entry', 'exit']
    """

    draws = True

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'stochastic', 'exclusion', 'traffic', 'nonequilibrium')
    references = ('Derrida, Domany, and Mukamel 1992', 'Schütz 2001', 'Helbing 2001')
    until = 500
    dt = 1

    def __init__(
        self,
        n_sites: int = 200,
        *,
        p: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.7,
        ring: bool = False,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.ring = ring
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _hop(s: SimpleNamespace) -> Expression:
        """Return 1 where a particle hops to the empty next site, and 0 otherwise.

        The hop has probability `p`, with one draw per edge.
        """
        return s.source.occupied * (1 - s.target.occupied) * bernoulli('hop', s.source.p)

    @staticmethod
    def _enter(s: SimpleNamespace) -> Expression:
        """Return 1 where a particle enters the empty first site, and 0 otherwise.

        The entry has probability `alpha`.
        """
        return (1 - s.target.occupied) * bernoulli('entry', s.target.alpha)

    @staticmethod
    def _leave(s: SimpleNamespace) -> Expression:
        """Return 1 where the particle on the last site leaves, and 0 otherwise.

        The exit has probability `beta`.
        """
        return s.source.occupied * bernoulli('exit', s.source.beta)

    def _build_system(self) -> CompositeSystem:
        """Return the totally asymmetric simple exclusion process with parallel update.

        The process runs on an open line or a ring.
        """
        rng = np.random.default_rng(self.random_state)
        occupied = np.zeros(self.n_sites)
        occupied[rng.choice(self.n_sites, round(self.density * self.n_sites), replace=False)] = 1.0
        boundary = 'periodic' if self.ring else 'fixed'
        lane = GroupElement(
            'lane',
            element=ExclusionSiteElement,
            space=LatticeSpace(self.n_sites, boundary=boundary),
            values={'occupied': occupied, 'p': self.p, 'alpha': self.alpha, 'beta': self.beta},
            on='sites',
        )
        hops = TransferInteraction(
            'hops',
            source='lane',
            target='lane',
            state='occupied',
            amount=self._hop,
            flow='hops',
            offset=1,
            boundary=boundary,
        )
        if self.ring:
            return CompositeSystem(self.label, elements=[lane], interactions=[hops], time=DiscreteTime())
        first, last = 'lane[0]', f'lane[{self.n_sites - 1}]'
        entry = TransferInteraction(
            'entry',
            source=first,
            target=first,
            state=Source(),
            into='occupied',
            amount=self._enter,
        )
        leaving = TransferInteraction(
            'exit',
            source=last,
            target=last,
            state='occupied',
            into=Sink(),
            amount=self._leave,
        )
        return CompositeSystem(self.label, elements=[lane], interactions=[hops, entry, leaving], time=DiscreteTime())


class DiffusionLimitedAggregationSystem(OldBaseSystem):
    """Diffusion-limited aggregation on a periodic grid.

    Random walkers stick to a seed at the center.

    Walkers that touch the aggregate join it, so the tips of the aggregate catch most walkers and it grows into a
    branched fractal of dimension about 1.7. The system has the agent group `walkers`.

    Args:
        n_walkers:
            The number of walkers, the seed included.

        size:
            The side of the grid.

        random_state:
            The seed or generator of the initial cells of the walkers.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DiffusionLimitedAggregationSystem
        >>> DiffusionLimitedAggregationSystem(10, size=11).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/statistical_mechanics'
    topics = ('agents', 'aggregation', 'fractal', 'random walk')
    references = ('Witten and Sander 1981', 'Charbonneau 2017')
    until = 200
    dt = 1

    def __init__(
        self,
        n_walkers: int = 400,
        *,
        size: int = 41,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        middle = self.size // 2
        cells = rng.integers(0, self.size, (self.n_walkers, 2))
        cells[0] = middle
        stuck = np.zeros(self.n_walkers)
        stuck[0] = 1.0
        walkers = GroupElement(
            'walkers',
            element=WalkerElement,
            size=self.n_walkers,
            values={'stuck': stuck},
            location=[tuple(map(int, cell)) for cell in cells],
        )
        return CompositeSystem(
            self.label,
            elements=[walkers],
            space=LatticeSpace((self.size, self.size), boundary='periodic', metric='manhattan'),
        )
