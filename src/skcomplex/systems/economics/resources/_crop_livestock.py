"""Resources systems: a farming household that rears livestock.

Its wealth from crops buys livestock that it rears and sells.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, floor, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import lag, where
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class CropLivestockElement(Element):
    """Farming household with crop land, wealth, and livestock.

    Its purchases join the herd after a maturation delay.

    Every year the household sells `INT(livestock sale_fraction)` animals and buys `INT(wealth / purchase_price)`
    animals when that number is at most `max_livestock + sale - livestock`, and none otherwise. Its wealth grows by the
    crop income, the area times the normal yield times the yield multiplier times the price, and the livestock income,
    and falls by the purchases and the household consumption. Purchased animals join the herd three years later, and
    the yield multiplier regenerates and degenerates at fractions of itself.

    Args:
        label:
            The label of the household.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The wealth of the household.

        livestock:
            The animals of the herd.

        yield_multiplier:
            The multiplier of the normal crop yield.

        purchases:
            The animals bought in the previous year.

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

        yield_regeneration:
            The fraction by which the yield multiplier regenerates every year.

        cropping_intensity:
            The fraction by which the yield multiplier degenerates every year.

    Examples:
        >>> from skcomplex.systems.economics.resources import CropLivestockElement
        >>> CropLivestockElement('household', max_livestock=10.0).max_livestock
        10.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 10000.0,
        livestock: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        yield_multiplier: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        purchases: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), lags=2)] = 0.0,
        crop_area: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        yield_normal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.2,
        price: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 50000.0,
        unit_price: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 35000.0,
        purchase_price: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 15000.0,
        household_consumption: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 75000.0,
        max_livestock: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.0,
        sale_fraction: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.2,
        yield_regeneration: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        cropping_intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth
        self.livestock = livestock
        self.yield_multiplier = yield_multiplier
        self.purchases = purchases
        self.crop_area = crop_area
        self.yield_normal = yield_normal
        self.price = price
        self.unit_price = unit_price
        self.purchase_price = purchase_price
        self.household_consumption = household_consumption
        self.max_livestock = max_livestock
        self.sale_fraction = sale_fraction
        self.yield_regeneration = yield_regeneration
        self.cropping_intensity = cropping_intensity

    @staticmethod
    def get_sale(s: SimpleNamespace) -> Expression:
        """Return the animals sold in the year, `INT(livestock sale_fraction)`.

        Args:
            s:
                The namespace of the household.

        Returns:
            sale:
                The animals sold.
        """
        return floor(s.livestock * s.sale_fraction)

    @staticmethod
    def get_purchase(s: SimpleNamespace) -> Expression:
        """Return the animals bought in the year.

        They are the affordable ones when the herd has room for them.

        Args:
            s:
                The namespace of the household.

        Returns:
            purchase:
                The animals bought.
        """
        affordable = floor(s.wealth / s.purchase_price)
        room = s.max_livestock + CropLivestockElement.get_sale(s) - s.livestock
        return where(affordable <= room, affordable, 0)

    @staticmethod
    def get_crop_production(s: SimpleNamespace) -> Expression:
        """Return the crop production.

        It is the area times the normal yield times the yield multiplier.

        Args:
            s:
                The namespace of the household.

        Returns:
            production:
                The crop production, in tonnes per year.
        """
        return s.crop_area * s.yield_normal * s.yield_multiplier

    @staticmethod
    def get_income_from_crop(s: SimpleNamespace) -> Expression:
        """Return the income from the crop, the production times the price.

        Args:
            s:
                The namespace of the household.

        Returns:
            income:
                The income from the crop per year.
        """
        return CropLivestockElement.get_crop_production(s) * s.price

    @staticmethod
    def get_income_from_livestock(s: SimpleNamespace) -> Expression:
        """Return the income from the livestock, the animals sold times their price.

        Args:
            s:
                The namespace of the household.

        Returns:
            income:
                The income from the livestock per year.
        """
        return CropLivestockElement.get_sale(s) * s.unit_price

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the yearly steps of the stocks.

        The purchases join the herd three years later.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        sale, purchase = self.get_sale(s), self.get_purchase(s)
        income = self.get_income_from_crop(s) + self.get_income_from_livestock(s)
        expenditure = purchase * s.purchase_price + s.household_consumption
        multiplier = s.yield_multiplier * (1 + s.yield_regeneration - s.cropping_intensity)
        return [
            UpdateLaw(s.wealth, s.wealth + income - expenditure),
            UpdateLaw(s.livestock, s.livestock + lag(s.purchases, 2) - sale),
            UpdateLaw(s.yield_multiplier, multiplier),
            UpdateLaw(s.purchases, purchase),
        ]


class CropLivestockHouseholdSystem(BaseSystem):
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
        >>> CropLivestockHouseholdSystem(wealth=20000.0).prepare().system_.elements[0].wealth
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
                ('crop_production', CropLivestockElement.get_crop_production),
                ('income_from_crop', CropLivestockElement.get_income_from_crop),
                ('income_from_livestock', CropLivestockElement.get_income_from_livestock),
                ('sale', CropLivestockElement.get_sale),
                ('purchase', CropLivestockElement.get_purchase),
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
        super().__init__(label)
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

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        sale = int(self.livestock * self.sale_fraction)
        affordable = int(self.wealth / self.purchase_price)
        household = CropLivestockElement(
            'household',
            wealth=self.wealth,
            livestock=self.livestock,
            purchases=float(affordable if affordable <= self.max_livestock + sale - self.livestock else 0),
            crop_area=self.crop_area,
            yield_normal=self.yield_normal,
            price=self.price,
            unit_price=self.unit_price,
            purchase_price=self.purchase_price,
            household_consumption=self.household_consumption,
            max_livestock=self.max_livestock,
            sale_fraction=self.sale_fraction,
        )
        return System(self.label, elements=[household])
