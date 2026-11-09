"""Resources systems: a farming household that rears livestock.

Its wealth from crops buys livestock that it rears and sells.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, floor, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import lag, where
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


def _sale(s: SimpleNamespace) -> Expression:
    """Return the animals the household sells in the year, `INT(livestock sale_fraction)`."""
    return floor(s.livestock * s.sale_fraction)


def _purchase(s: SimpleNamespace) -> Expression:
    """Return the animals bought in the year, the affordable ones when the herd has room for them."""
    affordable = floor(s.wealth / s.purchase_price)
    room = s.max_livestock + _sale(s) - s.livestock
    return where(affordable <= room, affordable, 0)


def _crop_production(s: SimpleNamespace) -> Expression:
    """Return the crop production, the area times the normal yield times the yield multiplier."""
    return s.crop_area * s.yield_normal * s.yield_multiplier


def _income_from_crop(s: SimpleNamespace) -> Expression:
    """Return the income from the crop, the production times the price."""
    return _crop_production(s) * s.price


def _income_from_livestock(s: SimpleNamespace) -> Expression:
    """Return the income from the livestock, the animals sold times their price."""
    return _sale(s) * s.unit_price


class CropLivestockHouseholdSystem(DynamicalSystem):
    """Bala's crop livestock model.

    A farming household adds livestock rearing to its crop income. A household with a fixed crop area and a grant or
    loan to start rearing livestock earns the crop production times the price and the sales of animals, and spends on
    its consumption and on buying animals. Every year it sells the integer part of a fraction of its herd and buys as
    many animals as its wealth affords when the herd has room for them, and the animals it buys join the herd after
    three years. The equations and the defaults are those of the STELLA model of Bala et al. (2017, Section 4.11.6),
    whose `DELAY(purchase_of_livestock, 3)` is a pipeline of three years and whose `INT` truncates, so the model is a
    yearly map. The effects of the cropping intensity and the regeneration on the yield are zero. Livestock raises the
    income of the household considerably, but only after the maturation delay.

    The system has the stocks `household.wealth`, `household.livestock`, and `household.yield_multiplier`, and the
    purchases of the previous year `household.purchases`, over years. The observables of the system are
    `household.crop_production`, `household.income_from_crop`, `household.income_from_livestock`, `household.sale`, and
    `household.purchase`.

    Args:
        wealth:
            The initial wealth of the household.

        livestock:
            The initial herd.

        crop_area:
            The crop area, in hectares.

        yield_normal:
            The normal crop yield, in tonnes per hectare.

        price:
            The price of a tonne of crop.

        unit_price:
            The price an animal sells for.

        purchase_price:
            The price of an animal bought.

        household_consumption:
            The consumption of the household per year.

        max_livestock:
            The largest herd the household can rear.

        sale_fraction:
            The fraction of the herd sold every year.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import CropLivestockHouseholdSystem
        >>> CropLivestockHouseholdSystem(wealth=20000.0).prepare().system_.elements[0].get_value('wealth')
        20000.0
    """

    field = 'economics/resources'
    topics = (
        'agriculture',
        'livestock',
        'household',
        'livelihood',
        'stocks and flows',
        'system dynamics',
        'discrete map',
    )
    references = ('Bala et al. 2017',)
    until = 12
    observables = tuple(
        [
            ObservableSpec(name, function, element='household')
            for name, function in (
                ('crop_production', _crop_production),
                ('income_from_crop', _income_from_crop),
                ('income_from_livestock', _income_from_livestock),
                ('sale', _sale),
                ('purchase', _purchase),
            )
        ],
    )

    def __init__(
        self,
        wealth: float = 10000.0,
        livestock: float = 4.0,
        *,
        crop_area: float = 0.5,
        yield_normal: float = 3.2,
        price: float = 50000.0,
        unit_price: float = 35000.0,
        purchase_price: float = 15000.0,
        household_consumption: float = 75000.0,
        max_livestock: float = 8.0,
        sale_fraction: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.wealth = wealth
        self.livestock = livestock
        self.crop_area = crop_area
        self.yield_normal = yield_normal
        self.price = price
        self.unit_price = unit_price
        self.purchase_price = purchase_price
        self.household_consumption = household_consumption
        self.max_livestock = max_livestock
        self.sale_fraction = sale_fraction

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'household'

    def _get_time(self) -> str:
        """Return the discrete time of the yearly steps."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs, the first purchases those of the first year."""
        nonnegative = RealSet.closed_open(0, oo)
        sale = int(self.livestock * self.sale_fraction)
        affordable = int(self.wealth / self.purchase_price)
        purchases = float(affordable if affordable <= self.max_livestock + sale - self.livestock else 0)
        return {
            'wealth': DeclaredVariableSpec(self.wealth, StateSpec()),
            'livestock': DeclaredVariableSpec(self.livestock, StateSpec(domain=nonnegative)),
            'yield_multiplier': DeclaredVariableSpec(1.0, StateSpec(domain=nonnegative)),
            'purchases': DeclaredVariableSpec(purchases, StateSpec(domain=nonnegative, lags=2)),
            'crop_area': DeclaredVariableSpec(self.crop_area, ParameterSpec(domain=nonnegative)),
            'yield_normal': DeclaredVariableSpec(self.yield_normal, ParameterSpec(domain=nonnegative)),
            'price': DeclaredVariableSpec(self.price, ParameterSpec(domain=nonnegative)),
            'unit_price': DeclaredVariableSpec(self.unit_price, ParameterSpec(domain=nonnegative)),
            'purchase_price': DeclaredVariableSpec(self.purchase_price, ParameterSpec(domain=RealSet.open(0, oo))),
            'household_consumption': DeclaredVariableSpec(
                self.household_consumption, ParameterSpec(domain=nonnegative),
            ),
            'max_livestock': DeclaredVariableSpec(self.max_livestock, ParameterSpec(domain=nonnegative)),
            'sale_fraction': DeclaredVariableSpec(self.sale_fraction, ParameterSpec(domain=RealSet.closed(0, 1))),
            'yield_regeneration': DeclaredVariableSpec(0.0, ParameterSpec(domain=nonnegative)),
            'cropping_intensity': DeclaredVariableSpec(0.0, ParameterSpec(domain=nonnegative)),
        }

    def _get_equations(self) -> Equations:
        """Return the yearly steps of the stocks, the purchases joining the herd three years later."""
        return {
            'wealth': self._wealth,
            'livestock': lambda s: s.target.livestock + lag(s.target.purchases, 2) - _sale(s.target),
            'yield_multiplier': lambda s: (
                s.target.yield_multiplier * (1 + s.target.yield_regeneration - s.target.cropping_intensity)
            ),
            'purchases': lambda s: _purchase(s.target),
        }

    @staticmethod
    def _wealth(s: SimpleNamespace) -> Expression:
        """Return the wealth after the incomes and the spending of the year."""
        household = s.target
        income = _income_from_crop(household) + _income_from_livestock(household)
        expenditure = _purchase(household) * household.purchase_price + household.household_consumption
        return household.wealth + income - expenditure
