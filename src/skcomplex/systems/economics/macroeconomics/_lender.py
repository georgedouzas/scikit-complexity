"""Bank that lends up to a multiple of its equity at a rate that falls with its equity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._firm import FirmElement
from ._firm_view import FirmView
from ._lender_view import LenderView


class LenderElement(Element):
    """Bank that lends up to a multiple of its equity at a rate that falls with its equity.

    The bank asks the rate `alpha * equity ** -alpha` of Delli Gatti et al. (2010), so better capitalised banks lend
    cheaper, and it offers loans up to `equity / capital_ratio`. The maps of its settlement read the firms bound to
    their role `sources`, whose interest and defaults reach the bank they name in their reference state `partner`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        equity:
            The initial equity.

        ask:
            The interest rate the bank asks.

        supply:
            The loans the bank can still grant in the current step.

        lent:
            The loans granted in the current step.

        losses:
            The loans written off in the current step.

        alpha:
            The elasticity of the interest rate to the equity.

        capital_ratio:
            The smallest ratio of equity to loans.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import LenderElement
        >>> GroupElement('banks', element=LenderElement, size=5, values={'equity': 100.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        equity: Annotated[float, StateSpec()] = 100.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lent: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        losses: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        capital_ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.equity = equity
        self.ask = ask
        self.supply = supply
        self.lent = lent
        self.losses = losses
        self.alpha = alpha
        self.capital_ratio = capital_ratio

    @classmethod
    def _get_view(cls) -> type[LenderView]:
        """Return the view of a bank, with its equity, quote, and loans."""
        return LenderView

    @staticmethod
    def get_quote() -> tuple[tuple[str, Callable[[LenderView, SystemView], float]], ...]:
        """Return the states a bank sets in the plan of a step, its rate and lending capacity from its equity.

        Returns:
            quote:
                The name of every state and the function of its value, of the bank and the system view.
        """
        return (
            ('ask', lambda agent, system: agent.alpha * max(agent.equity, 1e-6) ** -agent.alpha),
            ('supply', lambda agent, system: max(agent.equity, 0.0) / agent.capital_ratio),
            ('lent', lambda agent, system: 0.0),
            ('losses', lambda agent, system: 0.0),
        )

    @staticmethod
    def _get_borrowers(agent: LenderView, system: SystemView) -> list[FirmView]:
        """Return the firms bound to the role `sources` that borrowed from a bank in the step.

        The firms are grouped by their bank once per step of each settlement map, which reads but does not change them.
        """
        step = (system.time, system.round)
        cached = system.storage.get('borrowers')
        if not (isinstance(cached, tuple) and cached[0] == step):
            borrowers: dict[int, list[FirmView]] = {}
            for firm in system.agents('sources', FirmView):
                bank = firm.referenced('partner') if firm.loan > 0 else None
                if bank is not None:
                    borrowers.setdefault(bank.id, []).append(firm)
            cached = (step, borrowers)
            system.storage['borrowers'] = cached
        return list(cached[1].get(agent.id, []))

    @staticmethod
    def get_income(agent: LenderView, system: SystemView) -> float:
        """Return the interest a bank earns in the step, less what its defaulting borrowers cannot repay.

        Args:
            agent:
                The bank.

            system:
                The system view of the settlement.

        Returns:
            income:
                The change of the equity.
        """
        borrowers = LenderElement._get_borrowers(agent, system)
        return sum(firm.loan * firm.rate + min(FirmElement.get_worth(firm), 0.0) for firm in borrowers)

    @staticmethod
    def get_losses(agent: LenderView, system: SystemView) -> float:
        """Return the loans of a bank written off in the step.

        Args:
            agent:
                The bank.

            system:
                The system view of the settlement.

        Returns:
            losses:
                The change of the losses.
        """
        borrowers = LenderElement._get_borrowers(agent, system)
        return sum(max(-FirmElement.get_worth(firm), 0.0) for firm in borrowers)
