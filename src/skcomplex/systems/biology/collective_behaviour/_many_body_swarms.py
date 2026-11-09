"""Swarms whose alignment acts through pairs, triples, and quadruples of neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, exp, oo, sqrt, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws import RateLaw
from ....core.components.spaces import ContinuousSpace, RadiusNeighborhood
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

NEIGHBOURS = 10


class SwarmAgentElement(Element):
    """Self-propelled agent driven to the speed `sqrt(alpha / beta)`.

    The position `x` is a state of order 2, whose velocity `v` is the first derivative `x_d1`, and the agent alone
    accelerates by the Rayleigh self-propulsion `(alpha - beta |v|^2) v`.

    Args:
        label:
            The label of the agent.

        visual:
            The visual of the agent, or `None` for the default one.

        x:
            The position.

        x_d1:
            The velocity.

        alpha:
            The propulsion.

        beta:
            The friction at high speed.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SwarmAgentElement
        >>> SwarmAgentElement('agent', alpha=2.0).alpha
        2.0
    """

    law_classes: ClassVar[tuple[type, ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.alpha = alpha
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[RateLaw]:
        """Return the self-propulsion of the agent."""
        velocity = s.x_d1
        return [RateLaw(s.x, (s.alpha - s.beta * velocity.dot_product(velocity)) * velocity)]


class MorseAlignmentInteraction(CouplingInteraction):
    """Morse force and Cucker-Smale alignment between neighbours under a smooth cutoff.

    An agent `i` gains from every neighbour `j` within `R` the acceleration
    `f_c(rho) (U'(rho) r_hat + gamma exp(-rho / l_v) u)`, where `rho` is their distance, `r_hat` the unit vector from
    `i` to `j`, `u` the velocity of `j` relative to `i`, `U(rho) = C_r exp(-rho / l_r) - C_a exp(-rho / l_a)` the Morse
    potential, and `f_c(rho) = (1 - rho^2 / R^2)^2` the smooth cutoff.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents, whose position `x` is a state of order 2.

        R:
            The radius of the neighbourhood and of the cutoff.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        gamma:
            The strength of the alignment.

        l_v:
            The range of the alignment.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import MorseAlignmentInteraction
        >>> MorseAlignmentInteraction('pairs', target='agents', R=3.0).neighborhood
        RadiusNeighborhood(radius=3.0)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        R: float = 3.0,
        C_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        l_r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        C_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l_a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l_v: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            write='x',
            function=self._get_pair_term,
            neighborhood=RadiusNeighborhood(R),
        )
        self.R = R
        self.C_r = C_r
        self.l_r = l_r
        self.C_a = C_a
        self.l_a = l_a
        self.gamma = gamma
        self.l_v = l_v

    def _get_pair_term(self, s: SimpleNamespace) -> FreeModuleElement:
        """Return the Morse force and the alignment of one neighbour under the cutoff."""
        offset = s.other.x - s.member.x
        rho = sqrt(offset.dot_product(offset))
        slope = -s.C_r / s.l_r * exp(-rho / s.l_r) + s.C_a / s.l_a * exp(-rho / s.l_a)
        cutoff = (1 - rho**2 / self.R**2) ** 2
        return cutoff * (slope * offset / rho + s.gamma * exp(-rho / s.l_v) * (s.other.x_d1 - s.member.x_d1))


class FlankingInteraction(CouplingInteraction):
    """Three-body alignment of an agent with pairs of neighbours that flank it.

    An agent `i` gains from every pair `j < k` of its neighbours within `R` the acceleration
    `kappa_v g_ij g_ik (1 - r_hat_ij . r_hat_ik) / 2 (u_ij + u_ik) / 2`, where `r_hat_ij` is the unit vector from `i`
    to `j`, `u_ij` the velocity of `j` relative to `i`, and `g(rho) = (1 - rho^2 / R^2)^2 exp(-rho / l_3)` the weight of
    a neighbour at the distance `rho`. Two neighbours on opposite sides weigh most, and two on one side not at all.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents, whose position `x` is a state of order 2.

        R:
            The radius of the neighbourhood and of the cutoff.

        kappa_v:
            The strength of the flanking alignment.

        l_3:
            The range of the weight of a neighbour.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlankingInteraction
        >>> FlankingInteraction('flanking', target='agents').size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        R: float = 3.0,
        kappa_v: Annotated[float, ParameterSpec()] = 2.0,
        l_3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            write='x',
            function=self._get_flanking,
            mode='hyperedges',
            size=3,
            neighborhood=RadiusNeighborhood(R),
        )
        self.R = R
        self.kappa_v = kappa_v
        self.l_3 = l_3

    def _get_flanking(self, s: SimpleNamespace) -> FreeModuleElement:
        """Return the flanking alignment of one pair of neighbours."""
        weight, relative = SR(1), vector(SR, [0] * len(s.member.x))
        directions = []
        for other in s.others:
            offset = other.x - s.member.x
            rho = sqrt(offset.dot_product(offset))
            weight *= (1 - rho**2 / self.R**2) ** 2 * exp(-rho / s.l_3)
            directions.append(offset / rho)
            relative += other.x_d1 - s.member.x_d1
        cosine = directions[0].dot_product(directions[1])
        return s.kappa_v * weight * (1 - cosine) / 2 * relative / 2


class EncirclementInteraction(CouplingInteraction):
    """Four-body alignment of an agent with triples of neighbours that surround it.

    An agent `i` gains from every triple `j < k < l` of its neighbours within `R` the acceleration
    `kappa_4 g_ij g_ik g_il (1 - |r_hat_ij + r_hat_ik + r_hat_il|^2 / 9) (u_ij + u_ik + u_il) / 3`, where `r_hat_ij`
    is the unit vector from `i` to `j`, `u_ij` the velocity of `j` relative to `i`, and
    `g(rho) = (1 - rho^2 / R^2)^2 exp(-rho / l_3)` the weight of a neighbour at the distance `rho`. Three neighbours
    spread evenly around the agent weigh most.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents, whose position `x` is a state of order 2.

        R:
            The radius of the neighbourhood and of the cutoff.

        kappa_4:
            The strength of the encirclement alignment.

        l_3:
            The range of the weight of a neighbour.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import EncirclementInteraction
        >>> EncirclementInteraction('encirclement', target='agents').size
        4
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        R: float = 3.0,
        kappa_4: Annotated[float, ParameterSpec()] = 4.0,
        l_3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            write='x',
            function=self._get_encirclement,
            mode='hyperedges',
            size=4,
            neighborhood=RadiusNeighborhood(R),
        )
        self.R = R
        self.kappa_4 = kappa_4
        self.l_3 = l_3

    def _get_encirclement(self, s: SimpleNamespace) -> FreeModuleElement:
        """Return the encirclement alignment of one triple of neighbours."""
        zero = vector(SR, [0] * len(s.member.x))
        weight, directions, relative = SR(1), zero, zero
        for other in s.others:
            offset = other.x - s.member.x
            rho = sqrt(offset.dot_product(offset))
            weight *= (1 - rho**2 / self.R**2) ** 2 * exp(-rho / s.l_3)
            directions += offset / rho
            relative += other.x_d1 - s.member.x_d1
        return s.kappa_4 * weight * (1 - directions.dot_product(directions) / 9) * relative / 3


class ManyBodyAlignmentSwarmSystem(BaseSystem):
    """A swarm of self-propelled agents that align through pairs, triples, and quadruples of neighbours.

    The acceleration of an agent is the sum of terms of one body order each (Gutleb 2026, Eq. 2): the self-propulsion
    `(alpha - beta |v|^2) v`, a Morse force and Cucker-Smale alignment with every neighbour, a 3-body flanking
    alignment with every pair of neighbours weighted by `kappa_v`, and a 4-body encirclement alignment with every
    triple of neighbours weighted by `kappa_4`. The neighbours are the agents within `R`, found again at every
    evaluation, and the smooth cutoff `(1 - rho^2 / R^2)^2` weighs every neighbour at the distance `rho`. With
    `kappa_v = kappa_4 = 0` the swarm is the classical pairwise collective. The agents move in a periodic box that holds
    about ten neighbours around each. The system has the positions `agents.x` and the velocities `agents.x_d1`.

    Args:
        size:
            The number of agents.

        dimension:
            The dimension of the space, 2 or 3.

        alpha:
            The propulsion.

        beta:
            The friction at high speed.

        C_r:
            The strength of the repulsion.

        l_r:
            The range of the repulsion.

        C_a:
            The strength of the attraction.

        l_a:
            The range of the attraction.

        gamma:
            The strength of the pairwise alignment.

        l_v:
            The range of the pairwise alignment.

        l_3:
            The range of the weight of a neighbour in the 3-body and 4-body terms.

        kappa_v:
            The strength of the 3-body flanking alignment.

        kappa_4:
            The strength of the 4-body encirclement alignment.

        R:
            The radius of the neighbourhood and of the cutoff.

        random_state:
            The seed or generator of the initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ManyBodyAlignmentSwarmSystem
        >>> system = ManyBodyAlignmentSwarmSystem(12).prepare().system_
        >>> system.elements[0].prepare().size_, [part.label for part in system.interactions]
        (12, ['pairs', 'flanking', 'encirclement'])
    """

    field = 'biology/collective_behaviour'
    topics = ('swarms', 'many-body interactions', 'alignment', 'self-propulsion')
    references = ('Gutleb 2026', "D'Orsogna et al. 2006", 'Cucker and Smale 2007')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        size: int = 40,
        dimension: int = 2,
        *,
        alpha: float = 20.0,
        beta: float = 20.0,
        C_r: float = 2.0,
        l_r: float = 0.5,
        C_a: float = 1.0,
        l_a: float = 2.0,
        gamma: float = 1.0,
        l_v: float = 1.0,
        l_3: float = 1.0,
        kappa_v: float = 2.0,
        kappa_4: float = 4.0,
        R: float = 3.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.C_r = C_r
        self.l_r = l_r
        self.C_a = C_a
        self.l_a = l_a
        self.gamma = gamma
        self.l_v = l_v
        self.l_3 = l_3
        self.kappa_v = kappa_v
        self.kappa_4 = kappa_4
        self.R = R
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the agents and their interactions of every body order.

        The agents start near the sites of a square or cubic lattice in a periodic box, whose spacing gives each about
        ten neighbours, at the preferred speed with random headings.
        """
        rng = np.random.default_rng(self.random_state)
        spacing = (np.pi * self.R**2 / NEIGHBOURS) ** (1 / self.dimension)
        sides = int(np.ceil(self.size ** (1 / self.dimension)))
        sites = np.stack(np.meshgrid(*[np.arange(sides)] * self.dimension, indexing='ij'), axis=-1)
        sites = sites.reshape(-1, self.dimension)[: self.size]
        positions = spacing * (sites + rng.uniform(-0.25, 0.25, (self.size, self.dimension)))
        headings = rng.normal(size=(self.size, self.dimension))
        speed = np.sqrt(self.alpha / self.beta)
        velocities = speed * headings / np.linalg.norm(headings, axis=1, keepdims=True)
        agents = GroupElement(
            'agents',
            element=SwarmAgentElement,
            size=self.size,
            values={'x': positions, 'x_d1': velocities, 'alpha': self.alpha, 'beta': self.beta},
        )
        pairs = MorseAlignmentInteraction(
            'pairs',
            target='agents',
            R=self.R,
            C_r=self.C_r,
            l_r=self.l_r,
            C_a=self.C_a,
            l_a=self.l_a,
            gamma=self.gamma,
            l_v=self.l_v,
        )
        flanking = FlankingInteraction('flanking', target='agents', R=self.R, kappa_v=self.kappa_v, l_3=self.l_3)
        encirclement = EncirclementInteraction(
            'encirclement',
            target='agents',
            R=self.R,
            kappa_4=self.kappa_4,
            l_3=self.l_3,
        )
        return CompositeSystem(
            self.label,
            elements=[agents],
            interactions=[pairs, flanking, encirclement],
            space=ContinuousSpace(extent=(spacing * sides,) * self.dimension, boundary='periodic'),
        )
