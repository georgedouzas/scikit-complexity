"""Firm that borrows in proportion to its net worth, invests at a random return, and defaults on negative worth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._firm_view import FirmView


class FirmElement(Element):
    """Firm that borrows in proportion to its net worth, invests at a random return, and defaults on negative worth.

    Every step the firm asks for `leverage` times its net worth from its bank, invests its net worth and the loan at
    the return `mean_return + volatility * z`, and pays the interest. When its net worth becomes negative it repays
    what it can, the bank writes off the rest, and a new firm with the net worth `entry` replaces it.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        net_worth:
            The initial net worth.

        demand:
            The loan still wanted in the current step.

        bid:
            The highest interest rate the firm pays.

        loan:
            The loan of the current step.

        rate:
            The interest rate of the loan, zero without a loan.

        partner:
            The id of the bank of the firm, `-1` before its first loan.

        gross_return:
            The return on the capital of the last step.

        injected:
            The net worth of an entrant that replaced the firm in the last step, zero otherwise.

        defaulted:
            Whether the firm defaulted in the last step.

        leverage:
            The ratio of the loan wanted to the net worth.

        mean_return:
            The mean return on capital.

        volatility:
            The standard deviation of the return on capital.

        entry:
            The net worth of an entrant.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.macroeconomics import FirmElement
        >>> GroupElement('firms', element=FirmElement, size=10, values={'net_worth': 5.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        net_worth: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        loan: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rate: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        partner: Annotated[float, StateSpec(domain=ZZ, reference='banks')] = -1,
        gross_return: Annotated[float, StateSpec()] = 0.0,
        injected: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        defaulted: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        leverage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mean_return: Annotated[float, ParameterSpec()] = 0.03,
        volatility: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        entry: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.net_worth = net_worth
        self.demand = demand
        self.bid = bid
        self.loan = loan
        self.rate = rate
        self.partner = partner
        self.gross_return = gross_return
        self.injected = injected
        self.defaulted = defaulted
        self.leverage = leverage
        self.mean_return = mean_return
        self.volatility = volatility
        self.entry = entry

    @classmethod
    def _get_view(cls) -> type[FirmView]:
        """Return the view of a firm, with its balance sheet and its loan."""
        return FirmView

    @staticmethod
    def get_return(agent: FirmView, model: ModelView) -> float:
        """Return the return on the net worth and the loan invested, at a normal rate of return of at least -1.

        Args:
            agent:
                The firm.

            model:
                The model view of the investment.

        Returns:
            gross_return:
                The return of the step.
        """
        draw = agent.mean_return + agent.volatility * model.rng.standard_normal()
        return (agent.net_worth + agent.loan) * max(draw, -1.0)

    @staticmethod
    def get_worth(agent: FirmView) -> float:
        """Return the net worth after the return and the interest, negative for a firm that defaults.

        Args:
            agent:
                The firm.

        Returns:
            worth:
                The net worth before a default.
        """
        return agent.net_worth + agent.gross_return - agent.loan * agent.rate

    @staticmethod
    def get_plan() -> tuple[tuple[str, Callable[[FirmView, ModelView], float]], ...]:
        """Return the states a firm sets in the plan of a step, a loan request of `leverage` times its net worth.

        Returns:
            plan:
                The name of every state and the function of its value, of the firm and the model view.
        """
        return (
            ('demand', lambda agent, model: agent.leverage * agent.net_worth),
            ('loan', lambda agent, model: 0.0),
            ('rate', lambda agent, model: 0.0),
            ('defaulted', lambda agent, model: 0.0),
            ('injected', lambda agent, model: 0.0),
        )

    @staticmethod
    def get_default() -> tuple[tuple[str, Callable[[FirmView, ModelView], float]], ...]:
        """Return the states a firm sets after its return, those of an entrant for a firm that defaults.

        Returns:
            default:
                The name of every state and the function of its value, of the firm and the model view.
        """
        worth = FirmElement.get_worth
        return (
            ('injected', lambda agent, model: agent.entry if worth(agent) < 0 else agent.injected),
            ('defaulted', lambda agent, model: 1.0 if worth(agent) < 0 else agent.defaulted),
            ('partner', lambda agent, model: -1.0 if worth(agent) < 0 else agent.partner),
            ('net_worth', lambda agent, model: agent.entry if worth(agent) < 0 else worth(agent)),
        )
