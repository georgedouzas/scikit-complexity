"""Driven lattice particles.

The module covers zero-range processes, the energy cascade on a tree, and exclusion with open boundaries.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import BaseInteraction, TransferInteraction
from ....core.components.laws import Source
from ....core.components.laws.utils import where
from ....core.components.spaces import ContinuousSpace, LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.backends import NumbaBackend
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class ParticleSiteElement(Element):
    """Site of a lattice of particles, holding the number `n` of particles or quanta.

    Args:
        label:
            The label of the site or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of particles.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ParticleSiteElement
        >>> ParticleSiteElement('site', n=3.0).n
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class TreeNodeElement(ParticleSiteElement):
    """Node of a tree that holds a number `n` of energy quanta.

    The node is drawn at a fixed position.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of quanta.

        x:
            The position of the node in the drawing of the tree.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TreeNodeElement
        >>> TreeNodeElement('root').n
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'points',
            {'position': 'x', 'radius': 0.04, 'color_by': 'n', 'color_range': (0.0, 2.0), 'states': (), 'trail': 0},
        ),
        *,
        n: float = 0.0,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
        self.x = x


class ExclusionHopsInteraction(TransferInteraction):
    """Hops of a particle from every site of a line to an empty neighbour.

    The particle hops to the right at rate `p` and to the left at `q`.

    No particle hops past the ends of the line. It is a transfer of one particle along every edge of the line, to the
    right and back, from an occupied site to an empty one.

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
        >>> from skcomplex.systems.physics.statistical_mechanics import ExclusionHopsInteraction
        >>> ExclusionHopsInteraction('hops', sites='lane').p
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(
            label,
            source=sites,
            target=sites,
            state='n',
            amount=1.0,
            rate=self._get_right_rate,
            reverse_rate=self._get_left_rate,
            offset=1,
            boundary='fixed',
        )
        self.sites = sites
        self.p = p
        self.q = q

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the sites, whose every member hops to the next one and back."""
        return [(namespace.sites, namespace.sites)]

    @staticmethod
    def _get_right_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `p` of a hop to the right from an occupied site to an empty one."""
        return s.p * s.source.n * (1 - s.target.n)

    @staticmethod
    def _get_left_rate(s: SimpleNamespace) -> Expression:
        """Return the rate of a hop to the left, `q` from an occupied site to an empty one."""
        return s.q * s.target.n * (1 - s.source.n)


class ReservoirInteraction(TransferInteraction):
    """Reservoir that adds and removes particles at its sites.

    The reservoir adds one particle to each of its sites at rate `inflow` and removes one at rate `outflow`.

    A particle leaves only an occupied site. With exclusion a particle enters only an empty site. It is a transfer of
    one particle from the environment into every site, with jumps back to the environment at the outflow rate.

    Args:
        label:
            The label of the reservoir.

        sites:
            The sites in contact with the reservoir.

        inflow:
            The rate at which a particle enters a site.

        outflow:
            The rate at which a particle leaves an occupied site.

        exclusion:
            Whether a site holds at most one particle.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ReservoirInteraction
        >>> ReservoirInteraction('bath', sites=['lane[0]'], inflow=0.5).outflow
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[list[str], RoleSpec(variadic=True)],
        inflow: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        outflow: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        exclusion: bool = False,
    ) -> None:
        super().__init__(
            label,
            source='',
            target='',
            state=Source(),
            into='n',
            amount=1.0,
            rate=self._get_entry_rate,
            reverse_rate=self._get_exit_rate,
        )
        self.sites = sites
        self.inflow = inflow
        self.outflow = outflow
        self.exclusion = exclusion

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return every site as the target of its own entries."""
        return [(site, site) for site in namespace.sites]

    def _get_entry_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate at which a particle enters a site.

        Under exclusion a particle enters only an empty site.
        """
        return s.inflow * (1 - s.target.n if self.exclusion else 1)

    @staticmethod
    def _get_exit_rate(s: SimpleNamespace) -> Expression:
        """Return the rate at which a particle leaves an occupied site."""
        return s.outflow * where(s.target.n > 0, 1, 0)


class ZeroRangeHopInteraction(TransferInteraction):
    """Hop of one particle along a link.

    The particle hops forward at rate `p u(n)` and backward at rate `q u(n')`.

    The rate depends only on the occupation of the departure site, through `u(n) = 1 + b / n`. It is a transfer of one
    particle from the first site to the second, with jumps back at the reverse rate.

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
        >>> from skcomplex.systems.physics.statistical_mechanics import ZeroRangeHopInteraction
        >>> ZeroRangeHopInteraction('hop', link=['sites[0]', 'sites[1]']).b
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        link: Annotated[list[str], RoleSpec(variadic=True)],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
    ) -> None:
        super().__init__(
            label,
            source='',
            target='',
            state='n',
            amount=1.0,
            rate=self._get_forward_rate,
            reverse_rate=self._get_backward_rate,
        )
        self.link = link
        self.p = p
        self.q = q
        self.b = b

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the two sites of the link.

        They are the departure and the arrival of a forward hop.
        """
        first, second = namespace.link
        return [(first, second)]

    @staticmethod
    def _get_forward_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `p u(n)` of a hop from the first site."""
        return s.p * ZeroRangeHopInteraction._departure_rate(s.source.n, s.b)

    @staticmethod
    def _get_backward_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `q u(n')` of a hop from the second site."""
        return s.q * ZeroRangeHopInteraction._departure_rate(s.target.n, s.b)

    @staticmethod
    def _departure_rate(n: Expression, b: Expression) -> Expression:
        """Return the zero-range rate `u(n) = 1 + b / n` of a site with `n` particles.

        The rate is 0 for an empty site.
        """
        return where(n > 0, 1 + b / where(n > 0, n, 1), 0)


class ZeroRangeHopsInteraction(TransferInteraction):
    """Hops of one particle from every site of a ring.

    The particle hops to the right at rate `p u(n)` and to the left at rate `q u(n)`.

    The rate depends only on the occupation `n` of the departure site, through `u(n) = 1 + b / n`. It is a transfer of
    one particle along every edge of the ring, to the right and back.

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
        >>> from skcomplex.systems.physics.statistical_mechanics import ZeroRangeHopsInteraction
        >>> ZeroRangeHopsInteraction('hops', sites='sites').b
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
    ) -> None:
        super().__init__(
            label,
            source=sites,
            target=sites,
            state='n',
            amount=1.0,
            rate=self._get_right_rate,
            reverse_rate=self._get_left_rate,
            offset=1,
        )
        self.sites = sites
        self.p = p
        self.q = q
        self.b = b

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the sites, whose every member hops to the next one and back."""
        return [(namespace.sites, namespace.sites)]

    @staticmethod
    def _get_right_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `p u(n)` of a hop to the right."""
        return s.p * ZeroRangeHopInteraction._departure_rate(s.source.n, s.b)

    @staticmethod
    def _get_left_rate(s: SimpleNamespace) -> Expression:
        """Return the rate `q u(n)` of a hop to the left, from the site on the right."""
        return s.q * ZeroRangeHopInteraction._departure_rate(s.target.n, s.b)


TREE_ASPECT = 0.7


class ZeroRangeProcessSystem(BaseSystem):
    """The zero-range process on a ring.

    The process condenses above the critical density `1 / (b - 2)` for `b > 2`.

    A particle leaves a site with `n` particles at rate `u(n) = 1 + b / n`, to the right with weight `p` and to the left
    with weight `q`. The stationary distribution factorizes with the weights `f(n) = prod_k 1 / u(k)`, which decay as
    `n^-b`. Above the critical density a fluid at density `1 / (b - 2)` coexists with one site that holds the excess
    particles. The system has the occupations `sites.n`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZeroRangeProcessSystem
        >>> system = ZeroRangeProcessSystem(10, density=2.0).prepare().system_
        >>> float(system.elements[0].values['n'].sum()), len(system.interactions)
        (20.0, 1)
    """

    field = 'physics/statistical_mechanics'
    topics = ('zero-range process', 'condensation', 'nonequilibrium', 'jump process')
    references = ('Evans and Hanney 2005', 'Bertin 2021')
    until = 2000.0
    dt = 10.0
    backend = NumbaBackend()

    def __init__(
        self,
        n_sites: int = 50,
        *,
        density: float = 3.0,
        b: float = 4.0,
        p: float = 1.0,
        q: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.density = density
        self.b = b
        self.p = p
        self.q = q
        self.random_state = random_state

    @staticmethod
    def _scatter(n_particles: int, n_sites: int, random_state: int | np.random.Generator | None) -> Array:
        """Return the occupations of sites that received a number of particles.

        The particles arrive one at a time on random sites.
        """
        sites = np.random.default_rng(random_state).integers(0, n_sites, n_particles)
        return np.bincount(sites, minlength=n_sites).astype(float)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        occupations = self._scatter(round(self.density * self.n_sites), self.n_sites, self.random_state)
        sites = LatticeElement(
            'sites',
            element=ParticleSiteElement,
            space=LatticeSpace(self.n_sites),
            values={'n': occupations},
        )
        return System(
            self.label,
            elements=[sites],
            interactions=[ZeroRangeHopsInteraction('hops', sites='sites', p=self.p, q=self.q, b=self.b)],
        )


class DissipativeZeroRangeSystem(BaseSystem):
    """The dissipative zero-range process on a tree.

    Energy is injected at the root and lost at the leaves.

    Level `j` of the tree has `m^(j - 1)` nodes, each with `m` branches to the next level. A quantum `epsilon0` of
    energy crosses a branch between levels `j` and `j + 1` at rate `nu_j = nu k_j^alpha`, `k_j = m^(j - 1)`, in either
    direction from an occupied node. A reservoir at the inverse temperature `beta_ext` exchanges quanta with the root,
    adding one at rate `nu exp(-beta_ext epsilon0)` and removing one at rate `nu`, and every leaf loses a quantum at the
    rate `dissipation`. The stationary distribution is a product of geometric laws `z_j^n`, with `z_j` the solution of
    the linear balance of the currents `nu_j (z_j - z_(j + 1))` at every level, and the injected flux is `nu
    (exp(-beta_ext epsilon0) - z_1)`. The system has the quanta `nodes.n` of the nodes in breadth-first order, from the
    root to the leaves.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DissipativeZeroRangeSystem
        >>> system = DissipativeZeroRangeSystem(3, branches=3).prepare().system_
        >>> system.elements[0].size, len(system.interactions)
        (13, 14)
    """

    field = 'physics/statistical_mechanics'
    topics = ('zero-range process', 'energy cascade', 'nonequilibrium', 'tree', 'jump process')
    references = ('Bertin 2021', 'Bertin 2005')
    until = 200.0
    dt = 1.0
    backend = NumbaBackend()

    def __init__(
        self,
        n_levels: int = 5,
        *,
        branches: int = 2,
        nu: float = 1.0,
        alpha: float = 0.5,
        beta_ext: float = 1.0,
        epsilon0: float = 1.0,
        dissipation: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_levels = n_levels
        self.branches = branches
        self.nu = nu
        self.alpha = alpha
        self.beta_ext = beta_ext
        self.epsilon0 = epsilon0
        self.dissipation = dissipation

    @staticmethod
    def _level_start(level: int, branches: int) -> int:
        """Return the index of the first node of a level of a tree.

        The levels are counted from level 1, in breadth-first order.
        """
        return (branches ** (level - 1) - 1) // (branches - 1)

    @staticmethod
    def _tree_layout(n_levels: int, branches: int) -> Array:
        """Return the drawing positions of the nodes of a tree in breadth-first order.

        The positions lie on ellipses around the root.
        """
        rings = []
        for level in range(1, n_levels + 1):
            width = branches ** (level - 1)
            radius = (level - 1) / max(n_levels - 1, 1)
            angles = 2 * np.pi * (np.arange(width) + 0.5) / width
            rings.append(radius * np.column_stack([np.cos(angles), TREE_ASPECT * np.sin(angles)]))
        return np.concatenate(rings)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        n_nodes = self._level_start(self.n_levels + 1, self.branches)
        nodes = GroupElement(
            'nodes',
            element=TreeNodeElement,
            size=n_nodes,
            values={'x': self._tree_layout(self.n_levels, self.branches)},
        )
        interactions: list[BaseInteraction] = []
        for level in range(1, self.n_levels):
            rate = self.nu * float(self.branches ** (level - 1)) ** self.alpha
            start, below = self._level_start(level, self.branches), self._level_start(level + 1, self.branches)
            for index in range(self.branches ** (level - 1)):
                for branch in range(self.branches):
                    child = below + self.branches * index + branch
                    link = [f'nodes[{start + index}]', f'nodes[{child}]']
                    interactions.append(
                        ZeroRangeHopInteraction(f'transfer_{start + index}_{child}', link=link, p=rate, q=rate, b=0.0),
                    )
        leaves = [f'nodes[{index}]' for index in range(self._level_start(self.n_levels, self.branches), n_nodes)]
        interactions += [
            ReservoirInteraction(
                'injection',
                sites=['nodes[0]'],
                inflow=self.nu * np.exp(-self.beta_ext * self.epsilon0),
                outflow=self.nu,
            ),
            ReservoirInteraction('dissipation', sites=leaves, outflow=self.dissipation),
        ]
        return System(self.label, elements=[nodes], interactions=interactions, space=ContinuousSpace(n_dim=2))


class ASEPSystem(BaseSystem):
    """The asymmetric simple exclusion process on an open line, in continuous time.

    A particle hops to the empty right neighbour at rate `p` and to the empty left neighbour at rate `q`, enters the
    empty first site at rate `alpha`, and leaves the last site at rate `gamma`, the exit rate that Bertin writes `gamma`
    and others write `beta`. For `p = 1` and `q = 0` the current is `alpha (1 - alpha)` in the low-density phase `alpha
    < 1/2, gamma > alpha`, `gamma (1 - gamma)` in the high-density phase `gamma < 1/2, gamma < alpha`, and `1/4` in the
    maximal-current phase `alpha, gamma > 1/2`. The system has the occupations `lane.n`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ASEPSystem
        >>> len(ASEPSystem(10).prepare().system_.interactions)
        3
    """

    field = 'physics/statistical_mechanics'
    topics = ('exclusion', 'nonequilibrium', 'phase transition', 'traffic', 'jump process')
    references = ('Derrida, Evans, Hakim, and Pasquier 1993', 'Bertin 2021')
    until = 500.0
    dt = 2.0
    backend = NumbaBackend()

    def __init__(
        self,
        n_sites: int = 100,
        *,
        alpha: float = 0.3,
        gamma: float = 0.7,
        p: float = 1.0,
        q: float = 0.0,
        density: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.alpha = alpha
        self.gamma = gamma
        self.p = p
        self.q = q
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the asymmetric simple exclusion process on an open line.

        The process runs in continuous time.
        """
        occupied = (np.random.default_rng(self.random_state).random(self.n_sites) < self.density).astype(float)
        lane = LatticeElement(
            'lane',
            element=ParticleSiteElement,
            space=LatticeSpace(self.n_sites, boundary='fixed'),
            values={'n': occupied},
        )
        interactions = [
            ExclusionHopsInteraction('hops', sites='lane', p=self.p, q=self.q),
            ReservoirInteraction('entry', sites=['lane[0]'], inflow=self.alpha, exclusion=True),
            ReservoirInteraction('exit', sites=[f'lane[{self.n_sites - 1}]'], outflow=self.gamma, exclusion=True),
        ]
        return System(self.label, elements=[lane], interactions=interactions)
