"""Innovation systems: the diffusion of a new product.

The product diffuses through innovators and imitators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import StockFlowSystem
from ...general.dynamics.types import FlowTable


class BassDiffusionSystem(StockFlowSystem):
    """The Bass diffusion system, people who adopt a new product.

    They adopt through advertising and word of mouth. A market of `N` people splits into potential adopters `P` and
    adopters `A`. Potential adopters adopt at the rate `(p + q A / N) P`, where the coefficient of innovation `p`
    captures adoption from external influence, such as advertising, and the coefficient of imitation `q` adoption from
    contact with adopters. People are conserved, so the adoption is a flow from `P` to `A`. With `q > p` the adoption
    rate rises to a peak at the time `log(q / p) / (p + q)` and the adopters follow an S-shaped curve. Starting without
    adopters, the closed form is `A(t) = N (1 - exp(-(p + q) t)) / (1 + (q / p) exp(-(p + q) t))`. The defaults are the
    mean coefficients that Sultan, Farley and Lehmann (1990) estimated over many products, with time in years.

    The system has the potential adopters `market.P`, the adopters `market.A`, and the adoption flow `market.adoption`
    in people per year, over years.

    Args:
        market:
            The market potential `N`, the number of people who eventually adopt.

        adopters:
            The initial adopters, a part of the market.

        p:
            The coefficient of innovation, per year.

        q:
            The coefficient of imitation, per year.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import BassDiffusionSystem
        >>> BassDiffusionSystem(market=500.0, adopters=10.0).prepare().system_.elements[0].get_value('P')
        490.0
    """

    field = 'economics/innovation'
    topics = ('diffusion', 'marketing', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bass 1969',)
    until = 20.0
    dt = 0.25

    def __init__(
        self,
        market: float = 1000.0,
        *,
        adopters: float = 0.0,
        p: float = 0.03,
        q: float = 0.38,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.market = market
        self.adopters = adopters
        self.p = p
        self.q = q

    @staticmethod
    def _get_adoption(s: SimpleNamespace) -> Expression:
        """Return the adoption rate `(p + q A / (P + A)) P` of the market.

        A transfer moves people within the market.
        """
        market = s.source
        return (market.p + market.q * market.A / (market.P + market.A)) * market.P

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks."""
        return 'market'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the stocks and the parameters with their specs."""
        return {
            'P': DeclaredVariableSpec(
                self.market - self.adopters,
                StateSpec(domain=RealSet.closed_open(0, oo), quantity='people'),
            ),
            'A': DeclaredVariableSpec(self.adopters, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')),
            'p': DeclaredVariableSpec(self.p, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'q': DeclaredVariableSpec(self.q, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_flows(self) -> FlowTable:
        """Return the flows between the stocks."""
        return {
            'adoption': ('P', 'A', self._get_adoption),
        }
