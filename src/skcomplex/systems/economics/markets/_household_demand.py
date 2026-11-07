"""Markets systems: the demand of heterogeneous households for a good as its price rises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system

DEMANDS = ('cobb-douglas', 'elasticity')


FOOD_SHARES = (0.15, 0.13, 0.12, 0.10, 0.08)


FOOD_ELASTICITIES = (-0.6, -0.6, -0.5, -0.4, -0.3)


class BudgetHouseholdElement(Element):
    """Household with a budget that buys a good at a price that rises by a share every step.

    With the demand `'cobb-douglas'` the household maximizes `A^alpha B^(1 - alpha)` against a composite good `B` of
    price 1, so it buys `A = alpha m / p` and spends the share `alpha` of its budget `m` on the good. With the demand
    `'elasticity'` it spends the budget share `alpha` at the initial price `p0` and its demand changes with its
    own-price elasticity, `A = (alpha m / p0) (p / p0)^elasticity`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        demand:
            The demand for the good.

        expenditure:
            The expenditure on the good.

        utility:
            The Cobb-Douglas utility of the good and the composite good bought.

        price:
            The price of the good.

        budget:
            The budget of the household.

        alpha:
            The budget share of the good, the exponent of the Cobb-Douglas utility.

        elasticity:
            The own-price elasticity of the demand.

        change:
            The relative rise of the price every step.

        initial_price:
            The price at the start.

        kind:
            The demand, `'cobb-douglas'` or `'elasticity'`.

    Raises:
        ValidationError:
            The demand is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import BudgetHouseholdElement
        >>> GroupElement('households', element=BudgetHouseholdElement, size=2, values={'budget': [50.0, 150.0]}).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        expenditure: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        utility: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        budget: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 100.0,
        alpha: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.1,
        elasticity: Annotated[float, StateSpec()] = -1.0,
        change: Annotated[float, ParameterSpec(domain=RealSet.open(-1, oo))] = 0.1,
        initial_price: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        kind: str = 'cobb-douglas',
    ) -> None:
        super().__init__(label, visual=visual)
        self.demand = demand
        self.expenditure = expenditure
        self.utility = utility
        self.price = price
        self.budget = budget
        self.alpha = alpha
        self.elasticity = elasticity
        self.change = change
        self.initial_price = initial_price
        self.kind = kind

    def _check(self) -> None:
        """Raise an error for an unknown demand.

        Raises:
            ValidationError:
                The demand is unknown.
        """
        super()._check()
        if self.kind not in DEMANDS:
            error_msg = f'The demand of {self.label!r} is {self.kind!r}. Use one of {DEMANDS}.'
            raise ValidationError(error_msg)

    def get_demand(self, s: SimpleNamespace, price: Expression) -> Expression:
        """Return the demand of the household at a price.

        Args:
            s:
                The namespace of the household.

            price:
                The price.

        Returns:
            demand:
                The demand for the good.
        """
        if self.kind == 'cobb-douglas':
            return s.alpha * s.budget / price
        return s.alpha * s.budget / s.initial_price * (price / s.initial_price) ** s.elasticity

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rise of the price and the demand, expenditure, and utility at the new price.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        price = s.price * (1 + s.change)
        demand = self.get_demand(s, price)
        rest = s.budget - price * demand
        return [
            UpdateLaw(s.price, price),
            UpdateLaw(s.demand, demand),
            UpdateLaw(s.expenditure, price * demand),
            UpdateLaw(s.utility, demand**s.alpha * rest ** (1 - s.alpha)),
        ]


def _budgets(n_households: int, minimum: float, rng: np.random.Generator) -> Array:
    """Return budgets drawn from an exponential law shifted right by the minimum, normalized to a mean of 100."""
    draws = minimum + rng.exponential(1.0, n_households) * (100.0 - minimum)
    return draws * 100.0 / draws.mean()


def _by_quintile(budgets: Array, means: Sequence[float], deviation: float, rng: np.random.Generator) -> Array:
    """Return normal draws around the mean of the budget quintile of every household."""
    quintiles = np.minimum((np.argsort(np.argsort(budgets)) * 5) // len(budgets), 4)
    return np.asarray(means, dtype=float)[quintiles] + deviation * rng.normal(size=len(budgets))


@register_system(
    field='economics/markets',
    tags=('demand', 'heterogeneity', 'agents', 'price elasticity', 'income distribution', 'discrete map'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 10},
    observables=lambda system: [
        ObservableSpec('total_demand', group='households', variable='demand', reduce='sum'),
        ObservableSpec('total_expenditure', group='households', variable='expenditure', reduce='sum'),
        ObservableSpec('mean_utility', group='households', variable='utility', reduce='mean'),
    ],
)
def household_demand(
    n_households: int = 1000,
    *,
    demand: str = 'cobb-douglas',
    shares: Sequence[float] = FOOD_SHARES,
    share_deviation: float = 0.02,
    elasticities: Sequence[float] = FOOD_ELASTICITIES,
    elasticity_deviation: float = 0.1,
    minimum: float = 100 / 3,
    change: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Hamill and Gilbert's heterogeneous households, whose demand for food traces a curve as its price rises.

    Every household has a budget drawn from an exponential law shifted right by a minimum and normalized to a mean of
    100, which gives a Gini coefficient of about a third and a P90/P10 ratio just over 4, and a budget share of food
    drawn around the mean of its budget quintile, from 15 percent for the poorest to 8 percent for the richest. Every
    step the price rises by the share `change`, from 1. With the Cobb-Douglas utility of Section 3.2 of Hamill and
    Gilbert (2016) the demand is `alpha m / p`, the own-price elasticity is -1, the expenditure stays the same, and the
    poorest lose the most utility. With the practical model of Section 3.3 the households also have own-price
    elasticities drawn around the means of their quintiles, from -0.6 to -0.3, so a price rise of 10 percent lowers
    the demand by about 4.4 percent and raises the expenditure by about 5.1 percent, more for the richest.

    Args:
        n_households:
            The number of households.

        demand:
            `'cobb-douglas'` for the utility model, `'elasticity'` for the practical model.

        shares:
            The mean budget share of the good of every budget quintile, from the poorest.

        share_deviation:
            The standard deviation of the budget shares within a quintile.

        elasticities:
            The mean own-price elasticity of every budget quintile, from the poorest.

        elasticity_deviation:
            The standard deviation of the elasticities within a quintile.

        minimum:
            The shift of the exponential law of the budgets before normalization.

        change:
            The relative rise of the price every step.

        random_state:
            The seed or generator of the budgets, shares, and elasticities.

    Returns:
        system:
            The system, with the group `households` of the demands `households.demand`, the expenditures
            `households.expenditure`, the utilities `households.utility`, and the prices `households.price`. The
            observables of the system are `total_demand`, `total_expenditure`, and `mean_utility`.

    Examples:
        >>> from skcomplex.systems.economics.markets import household_demand
        >>> household_demand(10).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    budgets = _budgets(n_households, minimum, rng)
    alphas = np.clip(_by_quintile(budgets, shares, share_deviation, rng), 0.0, 1.0)
    slopes = _by_quintile(budgets, elasticities, elasticity_deviation, rng) if demand == 'elasticity' else -1.0
    quantities = alphas * budgets
    households = GroupElement(
        'households',
        element=BudgetHouseholdElement,
        size=n_households,
        values={
            'budget': budgets,
            'alpha': alphas,
            'elasticity': slopes,
            'demand': quantities,
            'expenditure': quantities,
            'utility': quantities**alphas * (budgets - quantities) ** (1 - alphas),
            'change': change,
            'kind': demand,
        },
    )
    return System(elements=[households])
