"""Bank that creates money as loans and pays its interest income to a random person as a dividend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....engine.solvers import AgentView, ModelView
from ....specs import (
    StateSpec,
    VisualSpec,
)
from ._central_bank_view import CentralBankView


class CentralBankElement(Element):
    """Bank that creates money as loans and pays its interest income to a random person as a dividend.

    The transfer `dividends` of the model pays the income to a random person, or to a random firm when nobody lives.

    The money it issued, a negative holding, and the loans it holds sum to zero after every step, so every unit of
    money in the economy is matched by a loan.

    Args:
        label:
            The label of the bank or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        money:
            The money of the bank, the negative of the money it issued plus its income of the step.

        loans:
            The loans the bank holds.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.macroeconomics import CentralBankElement
        >>> GroupElement('bank', element=CentralBankElement, size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        loans: Annotated[float, StateSpec(quantity='credit')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money
        self.loans = loans

    @classmethod
    def _get_view(cls) -> type[CentralBankView]:
        """Return the view of the bank, with its money and loans."""
        return CentralBankView

    @staticmethod
    def get_income(agent: CentralBankView, holder: AgentView, model: ModelView) -> float:
        """Return the income of the step that the bank pays as a dividend, its money plus its loans.

        Args:
            agent:
                The bank.

            holder:
                The person, or the firm when nobody lives, that receives the dividend.

            model:
                The model view of the dividends.

        Returns:
            income:
                The income.
        """
        return agent.money + agent.loans
