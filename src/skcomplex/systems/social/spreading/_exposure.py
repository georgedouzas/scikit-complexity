"""Exposure contagion: SIS in the environments of a hypergraph.

The infected members give infective doses.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import xgi
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import count, hyperedges, total, uniform, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ParticipantElement(Element):
    """Participant in the environments of a contagion.

    It is susceptible at 0 and infected at 1. The infection and the recovery are an `ExposureInteraction` of the
    hypergraph, which `HypergraphExposureSisSystem` adds.

    Args:
        label:
            The label of the participant or the group.

        infected:
            The initial state, 0 or 1, one per participant or shared.

        recovery:
            The probability that an infected participant recovers in a step.

        space:
            The network with one participant per node, or `None` for one participant.

    Examples:
        >>> from skcomplex.systems.social.spreading import ParticipantElement
        >>> ParticipantElement('participant').recovery
        0.05
    """

    visual = VisualSpec('sites', {'states': ['infected'], 'low': 0.0, 'high': 1.0})

    def __init__(
        self,
        label: str,
        *,
        infected: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        recovery: Annotated[float, RealSet.closed(0, 1)] = 0.05,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.infected = infected
        self.recovery = recovery


class ExposureInteraction(Interaction):
    """Infection of participants by the doses of their environments, and recovery.

    The infected participants are susceptible, and the time is discrete. An environment, a hyperedge, gives each
    susceptible member the dose `rate * rho ** exponent`, where `rho` is the share of the other members that are
    infected, and the doses of the environments of a participant add up to `D`. In every step a susceptible participant
    is infected with probability `1 - exp(-D)`, and an infected one recovers with probability `recovery`, a parameter of
    the participants.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the participants on the network.

        rate:
            The dose of an environment whose other members are all infected.

        exponent:
            The exponent of the infection kernel, sublinear below 1 and supralinear above.

    Examples:
        >>> from skcomplex.systems.social.spreading import ExposureInteraction
        >>> ExposureInteraction('exposure', target='participants', exponent=1.5).exponent
        1.5
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        rate: Annotated[float, RealSet.closed_open(0, oo)] = 0.06,
        exponent: Annotated[float, RealSet.open(0, oo)] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate
        self.exponent = exponent

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next state of every participant, from the sum of the doses of its environments, as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            member = context.target
            others = total(member.members.infected, over=member.members) - member.infected
            share = others / (count(over=member.members) - 1)
            dose = hyperedges(context.rate * share**context.exponent)
            draw = uniform('exposure')
            infection = where(draw < 1 - exp(-dose), 1, 0)
            self.laws_: list[BaseLaw] = [
                UpdateLaw(member.infected, where(member.infected == 1, where(draw < member.recovery, 0, 1), infection)),
            ]


class HypergraphExposureSisSystem(BaseSystem):
    """The SIS contagion of St-Onge et al. in the environments of a hypergraph.

    Its kernel is nonlinear. Participants meet in environments, the hyperedges of a random hypergraph. In every step an
    environment gives a susceptible member the dose `rate * rho ** exponent` from the share `rho` of its other members
    that are infected, the member is infected with probability `1 - exp(-D)` of the sum `D` of its doses, and an
    infected participant recovers with probability `recovery`. Heterogeneous exposure times give the power law
    `rho ** exponent` to the infection kernel. With a sublinear or linear kernel the epidemic sets in continuously, and
    with a supralinear kernel, `exponent > 1`, the healthy state stays stable: an endemic state coexists with it above a
    critical rate, so a small seed dies out and a large one persists. In mean field the prevalence `I` is endemic where
    `k rate I ** (exponent - 1) (1 - I) = recovery` for `k` environments per participant.

    The system has the states `participants.infected` and the prevalence `prevalence`.

    Args:
        n_nodes:
            The number of participants.

        sizes:
            The sizes of the environments.

        hyperdegree:
            The mean number of environments of a participant, split evenly over the sizes, whose members are drawn at
            random.

        rate:
            The dose of an environment whose other members are all infected.

        exponent:
            The exponent of the infection kernel.

        recovery:
            The probability of recovery in a step.

        share:
            The share of infected participants at the start.

        seed:
            The seed of the hypergraph and the infected participants.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.spreading import HypergraphExposureSisSystem
        >>> len(HypergraphExposureSisSystem(40).prepare().system_.elements[0].infected)
        40
    """

    field = 'social/spreading'
    topics = ('contagion', 'hypergraph', 'exposure', 'bistability')
    references = ('St-Onge et al. 2021', 'Majhi et al. 2022')
    until = 200
    dt = 1
    observables = (Observable('prevalence', group='participants', variable='infected'),)

    def __init__(
        self,
        n_nodes: int = 1000,
        *,
        sizes: Sequence[int] = (10, 20, 30),
        hyperdegree: float = 5.0,
        rate: float = 0.06,
        exponent: float = 2.0,
        recovery: float = 0.05,
        share: float = 0.3,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.sizes = sizes
        self.hyperdegree = hyperdegree
        self.rate = rate
        self.exponent = exponent
        self.recovery = recovery
        self.share = share
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        hypergraph = xgi.Hypergraph()
        hypergraph.add_nodes_from(range(self.n_nodes))
        for size in self.sizes:
            number = round(self.n_nodes * self.hyperdegree / (len(self.sizes) * size))
            hypergraph.add_edges_from(rng.choice(self.n_nodes, size, replace=False).tolist() for _ in range(number))
        infected = (rng.permutation(self.n_nodes) < round(self.share * self.n_nodes)).astype(float)
        participants = ParticipantElement(
            'participants',
            infected=infected,
            recovery=self.recovery,
            space=NetworkSpace(xgi.to_graph(hypergraph), hyperedges=hypergraph),
        )
        exposure = ExposureInteraction('exposure', target='participants', rate=self.rate, exponent=self.exponent)
        self.system_ = System(self.label, elements=[participants], interactions=[exposure])
