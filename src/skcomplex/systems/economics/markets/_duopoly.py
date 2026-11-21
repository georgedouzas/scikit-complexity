"""Markets systems: duopolists that set their output from beliefs.

The beliefs are about their rival.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role

N_FIRMS = 2


class DuopolistView(AgentViewProtocol, Protocol):
    """Duopolist as the market rule sees it: its `output` and its marginal `cost`."""

    output: float
    cost: float


class DuopolistElement(Element):
    """Firm of a Cournot duopoly.

    It is described by its output and its constant marginal cost.

    Args:
        label:
            The label of the agent or group.

        output:
            The initial output.

        cost:
            The marginal cost.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import DuopolistElement
        >>> DuopolistElement('firms', output=[100.0, 100.0], size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        *,
        output: Annotated[object, RealSet.closed_open(0, oo)] = 50.0,
        cost: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.output = output
        self.cost = cost


class CournotAdjustmentInteraction(Interaction):
    """Simultaneous best responses of two firms to their beliefs.

    The beliefs are about the rival's last output. Under the inverse demand `P = a - b Q`, each firm sets
    `q_i = (a - c_i) / (2 b) - belief_j / 2`, where the belief is the rival's last output scaled by a factor drawn
    uniformly from `[1 - error, 1 + error]`. It is a rule of the system that sets every output from the outputs of the
    step before, so both firms respond to them.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of the two firms.

        a:
            The intercept of the inverse demand.

        b:
            The slope of the inverse demand.

        error:
            The largest relative error of a firm's belief about the rival's output.

    Examples:
        >>> from skcomplex.systems.economics.markets import CournotAdjustmentInteraction
        >>> CournotAdjustmentInteraction('market', target='firms', error=0.1).error
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: Annotated[float, RealSet.open(0, oo)] = 1500.0,
        b: Annotated[float, RealSet.open(0, oo)] = 10.0,
        error: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.error = error

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the responses of the firms of every step as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context, self.respond)]

    def respond(self, system: SystemView) -> None:
        """Set the output of every firm to its response to the outputs of the step before.

        Args:
            system:
                The system view of the market.
        """
        firms = cast(list[DuopolistView], system.get_agents('target'))
        responses = [self._get_response(firm, system) for firm in firms]
        for firm, response in zip(firms, responses, strict=True):
            firm.output = response

    def _get_response(self, agent: DuopolistView, system: SystemView) -> float:
        """Return the best response of a firm to its belief.

        The belief is about the last output of its rival.
        """
        a, b, error = (system.get_parameter(name) for name in ('a', 'b', 'error'))
        rival = next(firm for firm in cast(list[DuopolistView], system.get_agents('target')) if firm.id != agent.id)
        belief = rival.output * (1 + error * float(system.rng.uniform(-1.0, 1.0)))
        return max((a - agent.cost) / (2 * b) - belief / 2, 0.0)


def _get_price(s: SimpleNamespace) -> Expression:
    """Return the market price of the total output.

    It follows the inverse demand of the interaction `market`.
    """
    return s.market.a - s.market.b * s.total


class CournotDuopolySystem(BaseSystem):
    """Two Cournot duopolists that adjust their output to their beliefs.

    The beliefs are about each other. Without errors the outputs converge to the Nash equilibrium,
    `q_i = (a - 2 c_i + c_j) / (3 b)`, which is 50 units each for the defaults. Errors in the beliefs keep the total
    output fluctuating around it. The system has the group `firms`, the total output `total`, and the price `price`.

    Args:
        outputs:
            The initial outputs of the two firms.

        costs:
            The marginal costs of the two firms.

        a:
            The intercept of the inverse demand.

        b:
            The slope of the inverse demand.

        error:
            The largest relative error of a firm's belief about the rival's output.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import CournotDuopolySystem
        >>> CournotDuopolySystem(error=0.01).prepare().system_.interactions[0].error
        0.01
    """

    field = 'economics/markets'
    topics = ('market', 'oligopoly', 'agents')
    references = ('Hamill and Gilbert 2016', 'Cournot 1838')
    until = 30
    observables = (
        Observable('total', group='firms', reduce='sum', variable='output'),
        Observable('price', _get_price),
    )

    def __init__(
        self,
        outputs: Sequence[float] = (100.0, 100.0),
        costs: Sequence[float] = (0.0, 0.0),
        *,
        a: float = 1500.0,
        b: float = 10.0,
        error: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.outputs = outputs
        self.costs = costs
        self.a = a
        self.b = b
        self.error = error

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        firms = DuopolistElement('firms', output=list(self.outputs), cost=list(self.costs), size=N_FIRMS)
        self.system_ = System(
            self.label,
            elements=[firms],
            interactions=[CournotAdjustmentInteraction('market', target='firms', a=self.a, b=self.b, error=self.error)],
        )
