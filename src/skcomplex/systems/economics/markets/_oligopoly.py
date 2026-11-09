"""Static oligopoly equilibria, and the fictitious play of a pricing duopoly.

The pricing duopoly is discretised.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.base import BasePart
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ChoiceInteraction
from ....core.components.laws import EquationLaw
from ....core.components.spaces import DiscreteTime, StaticTime
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec
from ....core.types import Array
from ..common.learning import PayoffLearningInteraction


class CournotBestResponseElement(Element):
    """Quantities of two firms at the best responses of a Cournot game.

    Each firm holds a share of the other. The conditions are `alpha - 2 q_i - q_j = c_i / (1 + lambda_i)` and its mirror
    (Forrest 2018 Eq. 21.5), whose solution is `q_i = (alpha - 2 c_i / (1 + lambda_i) + c_j / (1 + lambda_j)) / 3` (Eq.
    21.6).

    Args:
        label:
            The label of the element.

        q_i:
            The initial guess of the quantity of the first firm, which the equations replace.

        q_j:
            The initial guess of the quantity of the second firm, which the equations replace.

        demand:
            The intercept `alpha` of the inverse demand.

        c_i:
            The unit cost of the first firm.

        c_j:
            The unit cost of the second firm.

        lambda_i:
            The cross holding of the first firm.

        lambda_j:
            The cross holding of the second firm.

    Examples:
        >>> from skcomplex.systems.economics.markets import CournotBestResponseElement
        >>> CournotBestResponseElement('duopoly', demand=8.0).demand
        8.0
    """

    law_classes = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        q_i: Annotated[float, StateSpec()] = 0.0,
        q_j: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, ParameterSpec()] = 10.0,
        c_i: Annotated[float, ParameterSpec()] = 1.0,
        c_j: Annotated[float, ParameterSpec()] = 2.0,
        lambda_i: Annotated[float, ParameterSpec()] = 0.0,
        lambda_j: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.q_i = q_i
        self.q_j = q_j
        self.demand = demand
        self.c_i = c_i
        self.c_j = c_j
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the two conditions of the best responses.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equations of the two quantities.
        """
        return [
            EquationLaw(s.demand - 2 * s.q_i - s.q_j == s.c_i / (1 + s.lambda_i), target=s.q_i),
            EquationLaw(s.demand - 2 * s.q_j - s.q_i == s.c_j / (1 + s.lambda_j), target=s.q_j),
        ]


class PricePointElement(Element):
    """Equilibrium distribution of prices at one price.

    It makes a firm indifferent among its prices. A firm that charges `P` sells to its loyal share `alpha`, and to the
    switchers `beta` when it undercuts all its `m - 1` rivals, each with the probability `1 - F(P)`, for the expected
    profit `P (alpha + beta (1 - F)^(m - 1))`. In equilibrium every price gives the profit `alpha` of selling to the
    loyal share at the highest price 1, which `F(P) = 1 - (alpha (1 - P) / (beta P))^(1 / (m - 1))` solves (Forrest 2018
    Lemma 20.1).

    Args:
        label:
            The label of the point or the group.

        distribution:
            The initial guess of `F(P)`, which the equations replace.

        price:
            The price `P`.

        alpha:
            The loyal share of every firm.

        beta:
            The share of switchers.

        n_firms:
            The number of firms `m`.

    Examples:
        >>> from skcomplex.systems.economics.markets import PricePointElement
        >>> PricePointElement('point', price=0.5).price
        0.5
    """

    law_classes = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        distribution: Annotated[float, StateSpec()] = 0.5,
        price: Annotated[float, StateSpec(domain=RealSet.open_closed(0, 1))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        n_firms: Annotated[float, ParameterSpec(domain=RealSet.closed_open(2, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.distribution = distribution
        self.price = price
        self.alpha = alpha
        self.beta = beta
        self.n_firms = n_firms

    @staticmethod
    def get_profit(member: SimpleNamespace) -> Expression:
        """Return the expected profit of a firm at the price of a member.

        It serves the observables of a run.

        Args:
            member:
                The namespace of the states and parameters of the member.

        Returns:
            profit:
                The expected profit `P (alpha + beta (1 - F)^(m - 1))`.
        """
        return member.price * (member.alpha + member.beta * (1 - member.distribution) ** (member.n_firms - 1))

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation of the distribution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation of the distribution.
        """
        distribution = 1 - (s.alpha * (1 - s.price) / (s.beta * s.price)) ** (1 / (s.n_firms - 1))
        return [EquationLaw(s.distribution == distribution, target=s.distribution)]


class PricingFirmView(AgentView):
    """View of a firm of a pricing game, with its states as attributes."""

    price: float
    scores: Array


@dataclass(frozen=True, eq=False)
class PricingGameSpec(BaseSpec):
    """Grid of prices of two firms, each with a loyal share.

    The firms share the switchers who pay the lower price.

    Args:
        prices:
            The prices the firms choose among, increasing.

        alpha:
            The loyal share of every firm.

        beta:
            The share of switchers.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.economics.markets import PricingGameSpec
        >>> PricingGameSpec(np.array([0.5, 1.0]), 0.25, 0.5).beta
        0.5
    """

    prices: Array
    alpha: float
    beta: float

    def utility(self, agent: PricingFirmView, system: SystemView, option: float) -> float:
        """Return what a price has earned so far against the prices of the rival.

        It is its score.

        Args:
            agent:
                The firm.

            system:
                The system view, which the score does not read.

            option:
                The price.

        Returns:
            utility:
                The score of the price.
        """
        return float(agent.scores[int(np.searchsorted(self.prices, option))])

    def payoff(self, agent: PricingFirmView, system: SystemView, k: int) -> float:
        """Return what the price of index `k` earns against the rival's price.

        The rival's price is the one charged in this step.

        Args:
            agent:
                The firm.

            system:
                The system view, which holds the rival.

            k:
                The index of the price.

        Returns:
            payoff:
                The profit of the price.
        """
        rival = cast(PricingFirmView, next(item for item in system.agents(agent.group) if item.id != agent.id))
        price, against = float(self.prices[k]), float(rival.price)
        share = 1.0 if price < against else 0.5 if price == against else 0.0
        return price * (self.alpha + self.beta * share)


class PricingFirmElement(Element):
    """Firm that charges one price of a grid.

    It counts what every price would have earned.

    Args:
        label:
            The label of the firm or the group.

        price:
            The price the firm charges.

        scores:
            The sum of the payoffs of every price of the grid against the prices of the rival so far.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import PricingFirmElement
        >>> GroupElement('firms', element=PricingFirmElement, size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        price: Annotated[float, StateSpec()] = 1.0,
        scores: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.price = price
        self.scores = scores

    @classmethod
    def _get_view(cls) -> type[PricingFirmView]:
        """Return the view of the firms.

        It types their states.
        """
        return PricingFirmView


DYNAMICS = ('static', 'fictitious-play')
DUOPOLY = 2


def _price_observables(system: CompositeSystem) -> list[ObservableSpec]:
    """Return the lowest and the highest expected profit over the prices.

    Under learning, it returns the prices charged.
    """
    if isinstance(system.time, StaticTime):
        profit = PricePointElement.get_profit
        return [
            ObservableSpec('lowest_profit', profit, group='prices', reduce='min'),
            ObservableSpec('highest_profit', profit, group='prices', reduce='max'),
        ]
    return [
        ObservableSpec('lower_price', group='firms', variable='price', reduce='min'),
        ObservableSpec('higher_price', group='firms', variable='price', reduce='max'),
    ]


class OligopolyEquilibriaSystem(BaseSystem):
    """The equilibria of a Cournot duopoly and of a pricing game.

    The pricing game has firms with loyal customers and switchers. The static system solves the Cournot best responses
    (Forrest 2018 Eqs. 21.5-21.6) and, on a grid of prices, the symmetric mixed equilibrium of `m` firms of loyal share
    `alpha` and switchers `beta = 1 - m alpha`, whose price distribution makes a firm indifferent among its prices, at
    the profit `alpha` (Lemma 20.1). With `dynamics='fictitious-play'`, two firms learn to price on the grid instead:
    every round each plays a best response to the prices of its rival so far, and the prices they charge are distributed
    near the mixed equilibrium.

    The static system, with the states `cournot.q_i`, `cournot.q_j`, `prices.price`, and `prices.distribution` and the
    observables `lowest_profit` and `highest_profit`, or the learning system, with the group `firms` and the observables
    `lower_price` and `higher_price`. The catalog lists the observables, so a run passes them in `observables`.

    Args:
        n_firms:
            The number of firms of the pricing game, two under fictitious play.

        alpha:
            The loyal share of every firm.

        n_prices:
            The number of prices of the grid from `alpha / (alpha + beta)` to 1.

        demand:
            The intercept of the inverse demand of the Cournot duopoly.

        c_i:
            The unit cost of the first Cournot firm.

        c_j:
            The unit cost of the second Cournot firm.

        lambda_i:
            The cross holding of the first Cournot firm.

        lambda_j:
            The cross holding of the second Cournot firm.

        dynamics:
            `'static'` for the equilibria, or `'fictitious-play'` for the learning of two pricing firms.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The dynamics are unknown, or fictitious play has other than two firms.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.economics.markets import OligopolyEquilibriaSystem
        >>> trajectory = simulate(OligopolyEquilibriaSystem(), until=1.0)
        >>> round(float(trajectory.values('cournot.q_i')[0]), 6)
        3.333333
    """

    field = 'economics/markets'
    topics = ('oligopoly', 'equilibrium', 'static', 'game theory')
    references = ('Forrest 2018', 'Brown 1951')

    def __init__(
        self,
        n_firms: int = 2,
        *,
        alpha: float = 0.25,
        n_prices: int = 21,
        demand: float = 10.0,
        c_i: float = 1.0,
        c_j: float = 2.0,
        lambda_i: float = 0.0,
        lambda_j: float = 0.0,
        dynamics: str = 'static',
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.alpha = alpha
        self.n_prices = n_prices
        self.demand = demand
        self.c_i = c_i
        self.c_j = c_j
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.dynamics = dynamics

    @staticmethod
    def _fictitious_play(label: str, prices: Array, alpha: float, beta: float) -> CompositeSystem:
        """Return two firms that best respond to the payoffs of every price so far.

        The payoffs are summed against the rival.
        """
        game = PricingGameSpec(prices, alpha, beta)
        firms = GroupElement(
            'firms',
            element=PricingFirmElement,
            size=DUOPOLY,
            values={'price': np.full(DUOPOLY, prices[-1]), 'scores': np.zeros((DUOPOLY, prices.size))},
        )
        choice = ChoiceInteraction(
            'choice',
            target='firms',
            state='price',
            options=tuple(prices.tolist()),
            utility=game.utility,
            rule='best',
            form='agent',
            phase='choose',
        )
        beliefs = PayoffLearningInteraction(
            'beliefs',
            target='firms',
            scores='scores',
            payoff=game.payoff,
            rule='cumulative',
            form='agent',
            phase='learn',
        )
        schedule = PhasesScheduleSpec([PhaseSpec('choose'), PhaseSpec('learn')])
        return CompositeSystem(
            label,
            elements=[firms],
            interactions=[choice, beliefs],
            time=DiscreteTime(schedule=schedule),
        )

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.dynamics not in DYNAMICS or (self.dynamics == 'fictitious-play' and self.n_firms != DUOPOLY):
            error_msg = (
                f'OligopolyEquilibriaSystem has dynamics={self.dynamics!r} and n_firms={self.n_firms!r}. Use one of '
                f'{DYNAMICS}, and {DUOPOLY} firms under fictitious play.'
            )
            raise ValidationError(error_msg)
        beta = 1 - self.n_firms * self.alpha
        prices = np.linspace(self.alpha / (self.alpha + beta), 1.0, self.n_prices)
        if self.dynamics == 'fictitious-play':
            return self._fictitious_play(self.label, prices, self.alpha, beta)
        points = GroupElement(
            'prices',
            element=PricePointElement,
            size=self.n_prices,
            values={'price': prices, 'alpha': self.alpha, 'beta': beta, 'n_firms': float(self.n_firms)},
        )
        cournot = CournotBestResponseElement(
            'cournot',
            demand=self.demand,
            c_i=self.c_i,
            c_j=self.c_j,
            lambda_i=self.lambda_i,
            lambda_j=self.lambda_j,
        )
        return CompositeSystem(self.label, elements=[cournot, points], time=StaticTime())

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = _price_observables(self._pass_parameters())
        return settings
