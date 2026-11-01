"""Circuit of money between households, firms, and a bank, one period per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import TransferInteraction
from ....core.laws import Sink, Source
from ....specs import RoleSpec


class CircuitInteraction(TransferInteraction):
    """Circuit of money between households, firms, and a bank, one period per step.

    The flows of a period are the legs of one transfer between the deposits of the sectors, consumption, wages,
    interest, dividends, and profits, lending from the environment, and repayments to it, each reading the parameters
    of the sectors. Every leg is named by its flow, so the reporters `<label>.<flow>` record it.

    Args:
        label:
            The label of the circuit.

        households:
            The path of the households.

        firms:
            The path of the firms.

        bank:
            The path of the bank.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CircuitInteraction
        >>> CircuitInteraction('circuit', households='households', firms='firms', bank='bank').bank
        'bank'
    """

    def __init__(
        self,
        label: str,
        *,
        households: Annotated[str, RoleSpec()],
        firms: Annotated[str, RoleSpec()],
        bank: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=households,
            target=firms,
            state='households.deposits',
            into='firms.deposits',
            amount=lambda s: CircuitInteraction._consumption(s.households, s.firms),
            flow='consumption',
            legs=[
                ('firms.deposits', 'households.deposits', lambda s: s.firms.wage_share * s.firms.sales, 'wages'),
                ('firms.deposits', 'bank.deposits', lambda s: s.bank.interest * s.firms.loans, 'interest'),
                ('bank.deposits', 'households.deposits', lambda s: s.bank.payout * s.bank.deposits, 'dividends'),
                (
                    'firms.deposits',
                    'households.deposits',
                    lambda s: s.firms.distribution * s.firms.deposits,
                    'profits',
                ),
                (
                    Source('lending'),
                    'firms.deposits',
                    lambda s: CircuitInteraction._lending(s.firms, s.bank),
                    'lending',
                ),
                ('firms.deposits', Sink('repayment'), lambda s: s.firms.repayment * s.firms.loans, 'repayments'),
            ],
        )
        self.households = households
        self.firms = firms
        self.bank = bank

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the households and the firms, the first pair of the circuit, whose other legs name their roles."""
        return [(namespace.households, namespace.firms)]

    @staticmethod
    def _consumption(households: SimpleNamespace, firms: SimpleNamespace) -> Expression:
        """Return the consumption of a period, a share of the wages and a share of the deposits of the households."""
        return households.alpha_income * firms.wage_share * firms.sales + households.alpha_wealth * households.deposits

    @staticmethod
    def _lending(firms: SimpleNamespace, bank: SimpleNamespace) -> Expression:
        """Return the new loans of a period, in proportion to the sales of the firms."""
        return bank.credit * firms.sales
