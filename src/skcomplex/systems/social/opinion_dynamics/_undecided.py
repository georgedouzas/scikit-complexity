"""Undecided systems: opinions against, undecided, and for.

They live on a network and in the mean-field flows between them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Quantity
from ....core.components.interactions import MapInteraction, TransferInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.types import SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System


class UndecidedView(AgentView):
    """View of a voter of the three-state opinion system.

    Its states and parameters are attributes.
    """

    opinion: float
    sensitivity: float


class UndecidedElement(Element):
    """Voter who is against, undecided, or for, at -1, 0, or 1.

    It redraws its opinion from its neighbours.

    Args:
        label:
            The label of the voter or the group.

        opinion:
            The opinion, -1, 0, or 1.

        sensitivity:
            The sensitivity `beta` of the opinion to the sum of the opinions of the neighbours.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import UndecidedElement
        >>> UndecidedElement('voter').sensitivity
        2.0
    """

    visual = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (-1.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        opinion: Annotated[object, FiniteEnumeratedSet([-1, 0, 1])] = 0.0,
        sensitivity: Annotated[object, RealSet.closed_open(0, oo)] = 2.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.opinion = opinion
        self.sensitivity = sensitivity

    @classmethod
    def _get_view(cls) -> type[UndecidedView]:
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return UndecidedView


class UndecidedFlowsElement(Element):
    """Population split into the numbers `A` against, `B` for, and `C` undecided.

    Args:
        label:
            The label of the element.

        against:
            The number `A` of agents against.

        undecided:
            The number `C` of undecided agents.

        support:
            The number `B` of agents for.

        omega:
            The rate `omega` at which an undecided agent joins either side.

        gamma:
            The rate `gamma` at which an agent against becomes undecided.

        eta:
            The rate `eta` at which an agent for becomes undecided.

        drive:
            The exponent of `C` in the flows into the undecided group, 0 for constant rates and 1 for rates in
            proportion to the undecided.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import UndecidedFlowsElement
        >>> UndecidedFlowsElement('population').omega
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        against: Annotated[object, RealSet.closed_open(0, oo), Quantity('agents')] = 0.0,
        undecided: Annotated[object, RealSet.closed_open(0, oo), Quantity('agents')] = 100.0,
        support: Annotated[object, RealSet.closed_open(0, oo), Quantity('agents')] = 0.0,
        omega: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        gamma: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        eta: Annotated[object, RealSet.closed_open(0, oo)] = 0.02,
        drive: Annotated[object, FiniteEnumeratedSet([0, 1])] = 1.0,
    ) -> None:
        super().__init__(label)
        self.against = against
        self.undecided = undecided
        self.support = support
        self.omega = omega
        self.gamma = gamma
        self.eta = eta
        self.drive = drive


class ThreeStateOpinionSystem(BaseSystem):
    """The three-state opinion system on a random regular network.

    The agents are against, undecided, or for. Every step the agents redraw their opinions one after the other in random
    order, each from the sum `x` of the opinions of its neighbours: against with the probability `f(-beta x)`, for with
    `f(beta x)`, and undecided otherwise, where `f(t) = 1 / (2 (1 + exp(-t)))` (Jensen 2023 section 11.5, Project 1).
    Since `f(t) + f(-t) = 1 / 2`, half of the draws are undecided. In mean field, with every agent seeing `q <x>`, the
    mean opinion solves `<x> = tanh(beta q <x> / 2) / 2`, which has a non-zero solution only above `beta_c = 4 / q`. On
    the sparse network, domains of both signs and the undecided half keep the order weaker than in mean field.

    The system has the group `voters`.

    Args:
        n_agents:
            The number of agents.

        degree:
            The number `q` of neighbours of every agent.

        sensitivity:
            The sensitivity `beta` of the opinion to the sum of the opinions of the neighbours.

        random_state:
            The seed or generator of the network and of the initial opinions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ThreeStateOpinionSystem
        >>> ThreeStateOpinionSystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'network', 'undecided')
    references = ('Jensen 2023',)
    until = 50

    @staticmethod
    def _get_logistic_half(argument: float) -> float:
        """Return `f(t) = 1 / (2 (1 + exp(-t)))`, the probability of one side.

        Its values lie in (0, 1/2).
        """
        return float(0.5 / (1.0 + np.exp(-np.clip(argument, -500.0, 500.0))))

    def __init__(
        self,
        n_agents: int = 200,
        *,
        degree: int = 4,
        sensitivity: float = 2.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.degree = degree
        self.sensitivity = sensitivity
        self.random_state = random_state

    @staticmethod
    def _redraw(voter: UndecidedView, system: SystemViewProtocol) -> float:
        """Return the opinion a voter draws from the sum of the opinions of its neighbours."""
        total = sum(neighbor.opinion for neighbor in voter.get_neighbors() if isinstance(neighbor, UndecidedView))
        against = ThreeStateOpinionSystem._get_logistic_half(-voter.sensitivity * total)
        support = ThreeStateOpinionSystem._get_logistic_half(voter.sensitivity * total)
        draw = float(system.rng.random())
        if draw < against:
            return -1.0
        return 1.0 if draw < against + support else 0.0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        graph = nx.random_regular_graph(self.degree, self.n_agents, seed=int(rng.integers(2**31)))
        voters = UndecidedElement(
            'voters',
            opinion=rng.choice([-1.0, 0.0, 1.0], self.n_agents),
            sensitivity=self.sensitivity,
            size=self.n_agents,
        )
        redraw = MapInteraction(
            'redraw',
            target='voters',
            state='opinion',
            function=self._redraw,
            schedule='random-sequential',
            form='agent',
        )
        self.system_ = System(self.label, elements=[voters], interactions=[redraw], space=NetworkSpace(graph))


def _get_critical_size(s: SimpleNamespace) -> Expression:
    """Return the critical size `N_c = omega (1 / gamma + 1 / eta)`.

    It is the critical size of the flows driven by the undecided.
    """
    return s.omega * (1 / s.gamma + 1 / s.eta)


class UndecidedFlowsSystem(BaseSystem):
    """The mean-field flows between agents against, `A`, undecided, `C`, and for, `B`.

    Undecided agents join either side at the rate `omega`, and agents against and for become undecided at the rates
    `gamma` and `eta`, with no direct flow between the sides, so `N = A + B + C` is conserved (Jensen 2023 section
    11.5, Project 1). With constant rates, `driven=False`, `dA/dt = omega C - gamma A`, and the population settles at
    the attractive fixed point `A : B : C = omega / gamma : omega / eta : 1`. When the undecided drive the doubt,
    `driven=True`, the flows into the undecided grow with their number, `dA/dt = omega C - gamma A C`, and the
    dynamics change at `N_c = omega (1 / gamma + 1 / eta)`: above it `A = omega / gamma`, `B = omega / eta`, and
    `N - N_c` agents never make up their mind, and below it every agent ends on one side.

    The system has the numbers `population.against`, `population.undecided`, and `population.support`, and the flows
    `population.joining_against`, `population.joining_support`, `population.doubting_against`, and
    `population.doubting_support`. The observable of the system is `population.critical_size`.

    Args:
        n_agents:
            The number of agents, all undecided at the start.

        omega:
            The rate at which an undecided agent joins either side.

        gamma:
            The rate at which an agent against becomes undecided.

        eta:
            The rate at which an agent for becomes undecided.

        driven:
            Whether the flows into the undecided are in proportion to their number.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import UndecidedFlowsSystem
        >>> UndecidedFlowsSystem(driven=False).prepare().system_.elements[0].drive
        0.0
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'undecided', 'mean field', 'stocks and flows', 'ode')
    references = ('Jensen 2023',)
    until = 20.0
    dt = 0.1
    observables = (Observable('critical_size', _get_critical_size, element='population'),)

    def __init__(
        self,
        n_agents: float = 100.0,
        *,
        omega: float = 1.0,
        gamma: float = 0.05,
        eta: float = 0.02,
        driven: bool = True,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.omega = omega
        self.gamma = gamma
        self.eta = eta
        self.driven = driven

    @staticmethod
    def _get_joining_against(s: SimpleNamespace) -> Expression:
        """Return the flow of undecided agents against, `omega C`."""
        return s.source.omega * s.source.undecided

    @staticmethod
    def _get_joining_support(s: SimpleNamespace) -> Expression:
        """Return the flow of undecided agents for, `omega C`."""
        return s.source.omega * s.source.undecided

    @staticmethod
    def _get_doubting_against(s: SimpleNamespace) -> Expression:
        """Return the flow of agents against into the undecided, `gamma A C^drive`."""
        return s.source.gamma * s.source.against * s.source.undecided**s.source.drive

    @staticmethod
    def _get_doubting_support(s: SimpleNamespace) -> Expression:
        """Return the flow of agents for into the undecided, `eta B C^drive`."""
        return s.source.eta * s.source.support * s.source.undecided**s.source.drive

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        population = UndecidedFlowsElement(
            'population',
            undecided=self.n_agents,
            omega=self.omega,
            gamma=self.gamma,
            eta=self.eta,
            drive=1.0 if self.driven else 0.0,
        )
        flows = TransferInteraction(
            'flows',
            source='population',
            target='population',
            state='undecided',
            into='against',
            amount=self._get_joining_against,
            flow='joining_against',
            legs=[
                ('undecided', 'support', self._get_joining_support, 'joining_support'),
                ('against', 'undecided', self._get_doubting_against, 'doubting_against'),
                ('support', 'undecided', self._get_doubting_support, 'doubting_support'),
            ],
        )
        self.system_ = System(self.label, elements=[population], interactions=[flows])
