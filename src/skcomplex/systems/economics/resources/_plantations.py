"""Plantation systems: the area of a tree crop that expands and collapses.

It expands with its profit and collapses with its pests.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import BaseLaw, RateLaw, Sink
from ....core.components.laws.utils import clip, series
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

PROFITABILITY_EFFECT = (
    (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    (0.02, 0.738, 1.16, 1.5, 1.76, 2.0, 2.14, 2.21, 2.24, 2.24, 2.27),
)
YIELD_EFFECT = (
    (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    (0.645, 0.555, 0.47, 0.41, 0.37, 0.32, 0.29, 0.26, 0.23, 0.215, 0.21),
)
PROFIT_EFFECT = (
    (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0),
    (1.11, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02),
)
ECOLOGICAL_EFFECT = (
    (0.0, 8.33, 16.7, 25.0, 33.3, 41.7, 50.0, 58.3, 66.7, 75.0, 83.3, 91.7, 100.0),
    (1.0, 0.87, 0.725, 0.535, 0.41, 0.335, 0.24, 0.2, 0.11, 0.075, 0.045, 0.02, 0.005),
)
INTENSIFICATION_EFFECT = (
    (0.0, 8.33, 16.7, 25.0, 33.3, 41.7, 50.0, 58.3, 66.7, 75.0, 100.0),
    (1.04, 1.18, 1.24, 1.29, 1.34, 1.37, 1.41, 1.45, 1.46, 1.48, 1.48),
)
FULL_INDEX = 100.0


class PlantationElement(Element):
    """Cocoa land in forest, planted, and abandoned stocks.

    The shade is thinned and lets the insects in. Farmers plant forest towards a desired area that grows with the
    profitability of cocoa, abandon plantations whose yield and profit fall, and leave abandoned land to the forest or
    convert it to oil palm when cocoa pays less than palm. Removing shade raises the yield and lets the insects grow
    logistically, and the subsidy keeps the shade and controls the insects. The land flows are transfers within the
    plantation of the system.

    Args:
        label:
            The label of the plantation.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        forest:
            The initial forest land.

        area:
            The initial cocoa area.

        abandoned:
            The initial abandoned land.

        shade_removal:
            The initial index of shade removal, from 0 to 100.

        insects:
            The initial index of insect attack, from 0 to 100.

        yield_normal:
            The yield per area without insects or thinning.

        price:
            The farm price of cocoa.

        cost:
            The operating cost per area.

        thinning_cost:
            The cost per area of removing all shade.

        adjustment_time:
            The time to plant towards the desired area.

        abandon_time:
            The average time before a plantation is abandoned.

        conversion_time:
            The time abandoned land stays before it returns to forest or turns to oil palm.

        palm_margin:
            The profitability of oil palm, against which abandoned land converts.

        thinning_time:
            The time to remove the shade.

        insect_growth:
            The largest relative growth rate of the insects, reached with all shade removed.

        insect_control:
            The rate at which the subsidy controls the insects.

        subsidy:
            The coverage of the subsidy and extension programme, from 0 to 1.

    Examples:
        >>> from skcomplex.systems.economics.resources import PlantationElement
        >>> PlantationElement('cocoa', subsidy=0.5).subsidy
        0.5
    """

    law_classes = (RateLaw,)

    @staticmethod
    def _get_cocoa_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the thinning of shade and the growth and control of the insects."""
        unprotected = 1 - s.subsidy
        return [
            RateLaw(s.shade_removal, unprotected * (FULL_INDEX - s.shade_removal) / s.thinning_time),
            RateLaw(
                s.insects,
                s.insect_growth * unprotected * s.shade_removal / FULL_INDEX * s.insects * (1 - s.insects / FULL_INDEX)
                - s.insect_control * s.subsidy * s.insects,
            ),
        ]

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        forest: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 490.0,
        area: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 10.0,
        abandoned: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 0.0,
        shade_removal: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        insects: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        yield_normal: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        price: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        cost: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        thinning_cost: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        adjustment_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        abandon_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        conversion_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        palm_margin: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
        thinning_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        insect_growth: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        insect_control: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        subsidy: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.forest = forest
        self.area = area
        self.abandoned = abandoned
        self.shade_removal = shade_removal
        self.insects = insects
        self.yield_normal = yield_normal
        self.price = price
        self.cost = cost
        self.thinning_cost = thinning_cost
        self.adjustment_time = adjustment_time
        self.abandon_time = abandon_time
        self.conversion_time = conversion_time
        self.palm_margin = palm_margin
        self.thinning_time = thinning_time
        self.insect_growth = insect_growth
        self.insect_control = insect_control
        self.subsidy = subsidy

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the thinning of shade and the growth and control of the insects.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return PlantationElement._get_cocoa_laws(s)


def _get_cocoa_yield(s: SimpleNamespace) -> Expression:
    """Return the yield of cocoa per area.

    The insects and the removal of shade change it.
    """
    return s.yield_normal * series(s.insects, *ECOLOGICAL_EFFECT) * series(s.shade_removal, *INTENSIFICATION_EFFECT)


def _get_profit_per_area(s: SimpleNamespace) -> Expression:
    """Return the profit per area of cocoa land."""
    return s.price * _get_cocoa_yield(s) - s.cost - s.thinning_cost * s.shade_removal / FULL_INDEX


def _get_profitability(s: SimpleNamespace) -> Expression:
    """Return the profitability of cocoa land, the positive profit over the revenue."""
    return clip(_get_profit_per_area(s), 0) / (s.price * _get_cocoa_yield(s))


class CocoaSystem(BaseSystem):
    """A cocoa production system, plantations that boom and bust.

    They boom with their profit and bust with the insects they invite. The planted area moves towards a desired area,
    the current area times a table function of the profit margin, so a profitable crop expands it. Farmers thin the
    shading trees, which raises the yield through an intensification table but lets the insects multiply. Insect attack
    lowers the yield through an ecological table, and the area is abandoned at a rate that falls with the yield and the
    profit through two more tables. Abandoned land is converted to oil palm. A subsidy for shade conservation and pest
    control slows the thinning and controls the insects. Land is conserved, so the area booms while the yield is high
    and busts once the insects take over, unless a subsidy keeps the shade. The structure and the five table functions
    follow the boom and bust of cocoa in Malaysia of Bala, Arshad and Noh (2017, Chapter 8), and the rates, the price,
    and the costs are illustrative, with the area in thousands of hectares, the yield in hundreds of kilograms per
    hectare, the money in ringgit per hundred kilograms and per hectare, the shade removal and the insect attack as
    indices from 0 to 100, and time in years.

    The system has the land stocks `cocoa.area` and `cocoa.abandoned`, the indices `cocoa.shade_removal` and
    `cocoa.insects`, and the flows `cocoa.planting`, `cocoa.abandonment`, and `cocoa.conversion` per year, over years.
    The observables of the system are `cocoa.cocoa_yield`, `cocoa.profit_per_area`, and `cocoa.profitability`.

    Args:
        area:
            The initial planted area.

        price:
            The farm price of cocoa.

        cost:
            The operating cost per hectare.

        insect_growth:
            The largest relative growth rate of the insect attack, reached with all shade removed.

        subsidy:
            The coverage of the subsidy and extension programme, from 0 to 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import CocoaSystem
        >>> CocoaSystem(subsidy=1.0).prepare().system_.elements[0].subsidy
        1.0
    """

    field = 'economics/resources'
    topics = ('agriculture', 'boom and bust', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 50.0
    dt = 0.25
    observables = (
        ObservableSpec('cocoa_yield', _get_cocoa_yield, element='cocoa'),
        ObservableSpec('profit_per_area', _get_profit_per_area, element='cocoa'),
        ObservableSpec('profitability', _get_profitability, element='cocoa'),
    )

    @staticmethod
    def _get_planting(s: SimpleNamespace) -> Expression:
        """Return the planting of forest towards the desired area.

        The desired area grows with the profitability.
        """
        land = s.source
        desired = land.area * series(_get_profitability(land), *PROFITABILITY_EFFECT)
        return clip(desired - land.area, 0) / land.adjustment_time * land.forest / (land.forest + land.area)

    @staticmethod
    def _get_abandonment(s: SimpleNamespace) -> Expression:
        """Return the abandonment of plantations whose yield and profit fall."""
        land = s.source
        effects = series(_get_cocoa_yield(land), *YIELD_EFFECT) * series(_get_profit_per_area(land), *PROFIT_EFFECT)
        return land.area * effects / land.abandon_time

    @staticmethod
    def _get_conversion(land: SimpleNamespace) -> Expression:
        """Return the share of the abandoned land converted to oil palm.

        It grows as cocoa pays less than palm.
        """
        return clip(land.palm_margin - _get_profitability(land), 0) / land.palm_margin

    def __init__(
        self,
        area: float = 10.0,
        *,
        price: float = 40.0,
        cost: float = 100.0,
        insect_growth: float = 0.6,
        subsidy: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.area = area
        self.price = price
        self.cost = cost
        self.insect_growth = insect_growth
        self.subsidy = subsidy

    @staticmethod
    def _land_use() -> TransferInteraction:
        """Return the planting, abandonment, fallow, and conversion of cocoa land.

        They are reported under the plantation.
        """
        return TransferInteraction(
            'land_use',
            source='cocoa',
            target='cocoa',
            state='forest',
            into='area',
            amount=CocoaSystem._get_planting,
            flow='planting',
            legs=[
                ('area', 'abandoned', CocoaSystem._get_abandonment, 'abandonment'),
                (
                    'abandoned',
                    'forest',
                    lambda s: (1 - CocoaSystem._get_conversion(s.source))
                    * s.source.abandoned
                    / s.source.conversion_time,
                    'fallow',
                ),
                (
                    'abandoned',
                    Sink('oil_palm'),
                    lambda s: CocoaSystem._get_conversion(s.source) * s.source.abandoned / s.source.conversion_time,
                    'conversion',
                ),
            ],
        )

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        plantation = PlantationElement(
            'cocoa',
            area=self.area,
            price=self.price,
            cost=self.cost,
            insect_growth=self.insect_growth,
            subsidy=self.subsidy,
        )
        return CompositeSystem(self.label, elements=[plantation], interactions=[self._land_use()])
