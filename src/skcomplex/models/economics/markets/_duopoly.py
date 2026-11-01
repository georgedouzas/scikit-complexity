"""Markets models: duopolists that set their output from beliefs about their rival."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction
from ....core.system import System
from ....core.times import DiscreteTime
from ....engine.solvers import ModelView
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._duopolist_view import DuopolistView

N_FIRMS = 2


def _price(s: SimpleNamespace) -> Expression:
    """Return the market price of the total output under the inverse demand of the interaction `market`."""
    return s.market.a - s.market.b * s.total


class DuopolistElement(Element):
    """Firm of a Cournot duopoly, described by its output and its constant marginal cost.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        output:
            The initial output.

        cost:
            The marginal cost.

    Examples:
        >>> from skcomplex.models.economics.markets import DuopolistElement
        >>> GroupElement('firms', element=DuopolistElement, size=2, values={'output': [100.0, 100.0]}).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        output: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.output = output
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[DuopolistView]:
        """Return the view of a duopolist, with its output and cost."""
        return DuopolistView


class CournotAdjustmentInteraction(MapInteraction):
    """Simultaneous best responses of two firms to their beliefs about the rival's last output.

    Under the inverse demand `P = a - b Q`, each firm sets `q_i = (a - c_i) / (2 b) - belief_j / 2`, where the
    belief is the rival's last output scaled by a factor drawn uniformly from `[1 - error, 1 + error]`. It is a
    synchronous map of the agent form of the outputs, so both firms respond to the outputs of the step before.

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
        >>> from skcomplex.models.economics.markets import CournotAdjustmentInteraction
        >>> CournotAdjustmentInteraction('market', target='firms', error=0.1).check().error
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1500.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        error: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='output', function=self._get_response, form='agent')
        self.a = a
        self.b = b
        self.error = error

    def _get_response(self, agent: DuopolistView, model: ModelView) -> float:
        """Return the best response of a firm to its belief about the last output of its rival."""
        a, b, error = (model.parameter(name) for name in ('a', 'b', 'error'))
        rival = next(firm for firm in model.agents('target', DuopolistView) if firm.id != agent.id)
        belief = rival.output * (1 + error * float(model.rng.uniform(-1.0, 1.0)))
        return max((a - agent.cost) / (2 * b) - belief / 2, 0.0)


@register_model(
    field='economics/markets',
    tags=('market', 'oligopoly', 'agents'),
    references=('Hamill and Gilbert 2016', 'Cournot 1838'),
    defaults={'until': 30},
    observables=lambda system: [
        ObservableSpec('total', group='firms', reduce='sum', variable='output'),
        ObservableSpec('price', _price),
    ],
)
def cournot_duopoly(
    outputs: Sequence[float] = (100.0, 100.0),
    costs: Sequence[float] = (0.0, 0.0),
    *,
    a: float = 1500.0,
    b: float = 10.0,
    error: float = 0.0,
) -> System:
    """Return two Cournot duopolists that adjust their output to their beliefs about each other.

    Without errors the outputs converge to the Nash equilibrium, `q_i = (a - 2 c_i + c_j) / (3 b)`, which is 50 units
    each for the defaults. Errors in the beliefs keep the total output fluctuating around it.

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

    Returns:
        system:
            The model, with the group `firms`, the total output `total`, and the price `price`.

    Examples:
        >>> from skcomplex.models.economics.markets import cournot_duopoly
        >>> cournot_duopoly(error=0.01).interactions[0].error
        0.01
    """
    firms = GroupElement(
        'firms',
        element=DuopolistElement,
        size=N_FIRMS,
        values={'output': list(outputs), 'cost': list(costs)},
    )
    return System(
        elements=[firms],
        interactions=[CournotAdjustmentInteraction('market', target='firms', a=a, b=b, error=error)],
        time=DiscreteTime(),
    )
