"""Process systems: stocks of grain, waste, food, and water that engineered systems move between holders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import sin

from ....core.base import BaseLaw, Expression, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import RateLaw, Sink, Source, Symbols, TransferLaw, lookup, step, where
from ....core.specs import ObservableSpec
from ....core.system import System
from ....engine.runtime import register_model
from ._delays import delay3, smooth

MONTHS_PER_YEAR = 12.0
DAYS_PER_YEAR = 365.0
LANDFILL_ACCEPTANCE = ((0.0, 1.0), (0.9, 1.0), (1.0, 0.0))
WATER_STRESS = ((0.0, 0.0), (0.3, 0.5), (0.6, 1.0), (1.0, 1.0))
HUSK_SHARE = 0.2
BRAN_SHARE = 0.08
INITIAL_COLLECTION_SHARE = 0.75


def _smaller(first: Expression, second: Expression) -> Expression:
    """Return the smaller of two symbolic values."""
    return where(first < second, first, second)


def _positive(value: Expression) -> Expression:
    """Return the positive part of a symbolic value."""
    return where(value > 0, value, 0)


def _season(mean: Expression, amplitude: Expression, t: Expression, period: float) -> Expression:
    """Return a seasonal value that oscillates about its mean with a relative amplitude and a period."""
    return mean * (1 + amplitude * sin(2 * math.pi * t / period))


def _procurement(s: Symbols) -> Expression:
    """Return the procurement of paddy, the seasonal harvest after the procurement delay."""
    return delay3(_season(s.harvest, s.seasonality, s.t, MONTHS_PER_YEAR), s.procurement_delay)


def _indicated_milling(s: Symbols) -> Expression:
    """Return the indicated milling, the paddy milled over the milling time."""
    return s.paddy / s.milling_time


def _milling(s: Symbols) -> Expression:
    """Return the milling, the indicated milling limited by the capacity."""
    return _smaller(_indicated_milling(s), s.capacity)


def _rice_mill_laws(s: Symbols) -> list[BaseLaw]:
    """Return the procurement, milling, and sales of a rice mill, and the adjustment of its milling capacity."""
    milling = _milling(s)
    losses = 1 - s.outturn - s.husk_share - s.bran_share
    return [
        TransferLaw(Source('farms'), s.paddy, _procurement(s), label='paddy_purchases'),
        TransferLaw(s.paddy, s.rice, s.outturn * milling, label='milled_rice'),
        TransferLaw(s.paddy, s.husk, s.husk_share * milling, label='husk_output'),
        TransferLaw(s.paddy, s.bran, s.bran_share * milling, label='bran_output'),
        TransferLaw(s.paddy, Sink('losses'), losses * milling, label='milling_losses'),
        TransferLaw(s.rice, Sink('market'), s.rice / s.dispatch_time, label='rice_sales'),
        TransferLaw(s.husk, Sink('fuel'), s.husk / s.by_product_time, label='husk_sales'),
        TransferLaw(s.bran, Sink('feed'), s.bran / s.by_product_time, label='bran_sales'),
        RateLaw(s.capacity, (smooth(_indicated_milling(s), s.perception_time) - s.capacity) / s.expansion_time),
    ]


def _generation(s: Symbols) -> Expression:
    """Return the generation of waste by the population."""
    return s.per_capita * s.population


def _collection(s: Symbols) -> Expression:
    """Return the collection of waste, limited by the collection capacity."""
    return _smaller(s.uncollected / s.collection_time, s.collection_capacity)


def _acceptance(s: Symbols) -> Expression:
    """Return the share of the residual waste that the landfill accepts as it fills."""
    return lookup(s.landfill / s.landfill_capacity, LANDFILL_ACCEPTANCE)


def _waste_laws(s: Symbols) -> list[BaseLaw]:
    """Return the generation, collection, treatment, and disposal of the solid waste of a growing city."""
    generation, collection, acceptance = _generation(s), _collection(s), _acceptance(s)
    residual = (1 - s.recycling_share - s.composting_share) * collection
    return [
        TransferLaw(Source('births'), s.population, s.growth_rate * s.population, label='population_growth'),
        TransferLaw(Source('households'), s.uncollected, generation, label='waste_generation'),
        TransferLaw(s.uncollected, s.recycled, s.recycling_share * collection, label='recycling'),
        TransferLaw(s.uncollected, s.composted, s.composting_share * collection, label='composting'),
        TransferLaw(s.uncollected, s.landfill, acceptance * residual, label='landfilling'),
        TransferLaw(s.uncollected, s.dumped, (1 - acceptance) * residual, label='dumping'),
        TransferLaw(s.recycled, Sink('industry'), s.recycled / s.recycling_time, label='recycled_sales'),
        TransferLaw(s.composted, Sink('farms'), s.composted / s.composting_time, label='compost_sales'),
        RateLaw(s.collection_capacity, (generation - s.collection_capacity) / s.capacity_time),
    ]


def _need(s: Symbols) -> Expression:
    """Return the need of relief food, which starts at the onset of the disaster."""
    return s.ration * s.affected * step(s.t, s.onset)


def _shipments(s: Symbols) -> Expression:
    """Return the shipments that meet the need and close the gaps of the stock and of the pipeline."""
    need = _need(s)
    stock_gap = (s.cover * need - s.warehouse) / s.stock_time
    pipeline_gap = (s.shipping_delay * need - s.in_transit) / s.pipeline_time
    return _positive(need + stock_gap + pipeline_gap)


def _relief_laws(s: Symbols) -> list[BaseLaw]:
    """Return the shipments, arrivals, and distribution of relief food after a disaster."""
    shipments = _shipments(s)
    return [
        TransferLaw(Source('donors'), s.in_transit, shipments, label='shipping'),
        TransferLaw(s.in_transit, s.warehouse, delay3(shipments, s.shipping_delay), label='arrivals'),
        TransferLaw(
            s.warehouse,
            Sink('beneficiaries'),
            _smaller(_need(s), s.warehouse / s.distribution_time),
            label='distribution',
        ),
    ]


def _demand(s: Symbols) -> Expression:
    """Return the water demand of the crops, their evapotranspiration and the refill of the root zone."""
    return _positive(s.potential_et + (s.target * s.field_capacity - s.soil) / s.refill_time)


def _release(s: Symbols) -> Expression:
    """Return the release of the reservoir, the demand before the canal losses, limited by the reservoir."""
    return _smaller(_demand(s) / (1 - s.canal_loss), s.reservoir / s.release_time)


def _pumping(s: Symbols) -> Expression:
    """Return the pumping of groundwater, the demand the release leaves unmet, limited by the pumps and the aquifer."""
    unmet = _demand(s) - (1 - s.canal_loss) * _release(s)
    return _smaller(_positive(unmet), _smaller(s.pump_capacity, s.groundwater / s.pump_time))


def _irrigation_laws(s: Symbols) -> list[BaseLaw]:
    """Return the water balance of a reservoir, the root zone of the irrigated crops, and the aquifer below them."""
    release = _release(s)
    crop_water = s.potential_et * lookup(s.soil / s.field_capacity, WATER_STRESS)
    spill = _positive(s.reservoir - s.reservoir_capacity) / s.spill_time
    return [
        TransferLaw(
            Source('river'),
            s.reservoir,
            _season(s.river_inflow, s.seasonality, s.t, DAYS_PER_YEAR),
            label='inflow',
        ),
        TransferLaw(s.reservoir, Sink('atmosphere'), s.evaporation * s.reservoir, label='evaporation'),
        TransferLaw(s.reservoir, Sink('downstream'), spill, label='spill'),
        TransferLaw(s.reservoir, s.soil, (1 - s.canal_loss) * release, label='irrigation'),
        TransferLaw(s.reservoir, s.groundwater, s.canal_loss * release, label='seepage'),
        TransferLaw(s.groundwater, s.soil, _pumping(s), label='groundwater_pumping'),
        TransferLaw(s.soil, Sink('atmosphere'), crop_water, label='crop_water'),
        TransferLaw(s.soil, s.groundwater, s.percolation * s.soil, label='percolation'),
        TransferLaw(s.groundwater, Sink('downstream'), s.groundwater / s.baseflow_time, label='baseflow'),
    ]


class RiceMillElement(Element):
    """Rice mill element."""

    paddy = state(default=2000.0, domain=NumericDomain.NONNEGATIVE, quantity='grain')
    rice = state(default=1300.0, domain=NumericDomain.NONNEGATIVE, quantity='grain')
    husk = state(default=400.0, domain=NumericDomain.NONNEGATIVE, quantity='grain')
    bran = state(default=160.0, domain=NumericDomain.NONNEGATIVE, quantity='grain')
    capacity = state(default=2000.0, domain=NumericDomain.NONNEGATIVE)
    harvest = parameter(default=2000.0, domain=NumericDomain.NONNEGATIVE)
    seasonality = parameter(default=0.8, domain=NumericDomain.NONNEGATIVE)
    procurement_delay = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    milling_time = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    outturn = parameter(default=0.65, domain=NumericDomain.NONNEGATIVE)
    husk_share = parameter(default=HUSK_SHARE, domain=NumericDomain.NONNEGATIVE)
    bran_share = parameter(default=BRAN_SHARE, domain=NumericDomain.NONNEGATIVE)
    dispatch_time = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    by_product_time = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    perception_time = parameter(default=6.0, domain=NumericDomain.POSITIVE)
    expansion_time = parameter(default=12.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the rice mill element."""
        return _rice_mill_laws(s)


class WasteSystemElement(Element):
    """Waste system element."""

    population = state(default=1e6, domain=NumericDomain.NONNEGATIVE, quantity='people')
    uncollected = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='waste')
    recycled = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='waste')
    composted = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='waste')
    landfill = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='waste')
    dumped = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='waste')
    collection_capacity = state(default=1.5e5, domain=NumericDomain.NONNEGATIVE)
    growth_rate = parameter(default=0.02, domain=NumericDomain.REAL)
    per_capita = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)
    collection_time = parameter(default=0.05, domain=NumericDomain.POSITIVE)
    recycling_share = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)
    composting_share = parameter(default=0.3, domain=NumericDomain.NONNEGATIVE)
    landfill_capacity = parameter(default=2e6, domain=NumericDomain.POSITIVE)
    recycling_time = parameter(default=0.25, domain=NumericDomain.POSITIVE)
    composting_time = parameter(default=0.5, domain=NumericDomain.POSITIVE)
    capacity_time = parameter(default=3.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the waste system element."""
        return _waste_laws(s)


class ReliefPipelineElement(Element):
    """Relief pipeline element."""

    in_transit = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='food')
    warehouse = state(default=0.0, domain=NumericDomain.NONNEGATIVE, quantity='food')
    affected = parameter(default=1e5, domain=NumericDomain.NONNEGATIVE)
    ration = parameter(default=5e-4, domain=NumericDomain.NONNEGATIVE)
    onset = parameter(default=10.0, domain=NumericDomain.NONNEGATIVE)
    shipping_delay = parameter(default=14.0, domain=NumericDomain.POSITIVE)
    cover = parameter(default=7.0, domain=NumericDomain.NONNEGATIVE)
    stock_time = parameter(default=7.0, domain=NumericDomain.POSITIVE)
    pipeline_time = parameter(default=14.0, domain=NumericDomain.POSITIVE)
    distribution_time = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the relief pipeline element."""
        return _relief_laws(s)


class IrrigationSchemeElement(Element):
    """Irrigation scheme element."""

    reservoir = state(default=50.0, domain=NumericDomain.NONNEGATIVE, quantity='water')
    soil = state(default=12.0, domain=NumericDomain.NONNEGATIVE, quantity='water')
    groundwater = state(default=100.0, domain=NumericDomain.NONNEGATIVE, quantity='water')
    river_inflow = parameter(default=0.5, domain=NumericDomain.NONNEGATIVE)
    seasonality = parameter(default=0.9, domain=NumericDomain.NONNEGATIVE)
    reservoir_capacity = parameter(default=80.0, domain=NumericDomain.POSITIVE)
    evaporation = parameter(default=0.001, domain=NumericDomain.NONNEGATIVE)
    spill_time = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    release_time = parameter(default=10.0, domain=NumericDomain.POSITIVE)
    canal_loss = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)
    potential_et = parameter(default=0.4, domain=NumericDomain.NONNEGATIVE)
    field_capacity = parameter(default=20.0, domain=NumericDomain.POSITIVE)
    target = parameter(default=0.7, domain=NumericDomain.NONNEGATIVE)
    refill_time = parameter(default=5.0, domain=NumericDomain.POSITIVE)
    percolation = parameter(default=0.005, domain=NumericDomain.NONNEGATIVE)
    pump_capacity = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)
    pump_time = parameter(default=10.0, domain=NumericDomain.POSITIVE)
    baseflow_time = parameter(default=500.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the irrigation scheme element."""
        return _irrigation_laws(s)


@register_model(
    field='engineering/processes',
    tags=('agriculture', 'milling', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.25},
    observables=lambda system: [
        ObservableSpec('procurement', _procurement, element='mill'),
        ObservableSpec('indicated_milling', _indicated_milling, element='mill'),
        ObservableSpec('milling', _milling, element='mill'),
    ],
)
def rice_milling(
    *,
    harvest: float = 2000.0,
    seasonality: float = 0.8,
    procurement_delay: float = 1.0,
    outturn: float = 0.65,
    capacity: float = 2000.0,
    expansion_time: float = 12.0,
) -> System:
    """Return a rice milling system, a mill that buys paddy, mills it into rice and by-products, and sells them.

    Paddy arrives from the farms through a third-order procurement delay, at a rate that follows the harvest season
    around its mean. The mill processes its paddy stock in a month, as far as its milling capacity allows, and splits
    the milled paddy by mass into rice, the outturn, husk, bran, and milling losses. Rice, husk, and bran leave their
    inventories for the market at a fixed coverage. The capacity adjusts to the smoothed indicated milling rate, so in
    the harvest season the capacity lags and paddy accumulates, and in the lean season the capacity idles. Grain is
    conserved, every change of the paddy, rice, husk, and bran stocks is a flow, and over a year the milled paddy and
    the rice sales match the mean harvest and its outturn. The structure follows the rice milling case of Bala,
    Arshad and Noh (2017), with illustrative parameters in tonnes and months, since the case data are specific to
    their study.

    Args:
        harvest:
            The mean paddy procurement in tonnes per month.

        seasonality:
            The relative amplitude of the yearly harvest cycle.

        procurement_delay:
            The average time for purchased paddy to reach the mill, in months.

        outturn:
            The fraction of the milled paddy that becomes rice.

        capacity:
            The initial milling capacity in tonnes of paddy per month.

        expansion_time:
            The time for the capacity to adjust to the perceived milling rate, in months.

    Returns:
        system:
            The model, with the grain stocks `mill.paddy`, `mill.rice`, `mill.husk`, and `mill.bran` in tonnes, the
            capacity `mill.capacity`, and the flows `mill.paddy_purchases`, `mill.milled_rice`, `mill.husk_output`,
            `mill.bran_output`, `mill.milling_losses`, `mill.rice_sales`, `mill.husk_sales`, and `mill.bran_sales` in
            tonnes per month, over months. The observables of the model are `mill.procurement`,
            `mill.indicated_milling`, and `mill.milling`.

    Examples:
        >>> from skcomplex.models.engineering.processes import rice_milling
        >>> rice_milling(capacity=1500.0).elements[0].capacity
        1500.0
    """
    mill = RiceMillElement(
        'mill',
        paddy=harvest,
        rice=outturn * harvest,
        husk=HUSK_SHARE * harvest,
        bran=BRAN_SHARE * harvest,
        capacity=capacity,
        harvest=harvest,
        seasonality=seasonality,
        procurement_delay=procurement_delay,
        outturn=outturn,
        expansion_time=expansion_time,
    )
    return System(elements=[mill])


@register_model(
    field='engineering/processes',
    tags=('waste management', 'urban', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 40.0, 'dt': 0.1},
    observables=lambda system: [
        ObservableSpec('generation', _generation, element='city'),
        ObservableSpec('collection', _collection, element='city'),
        ObservableSpec('acceptance', _acceptance, element='city'),
    ],
)
def solid_waste(
    population: float = 1e6,
    *,
    growth_rate: float = 0.02,
    per_capita: float = 0.2,
    recycling_share: float = 0.2,
    composting_share: float = 0.3,
    landfill_capacity: float = 2e6,
) -> System:
    """Return a solid waste system, the waste of a growing city collected, recycled, composted, and landfilled.

    The population grows exponentially and generates waste in proportion to its size. The waste waits uncollected
    until the collection, limited by a collection capacity that the city expands towards the generation with a delay,
    removes it. The collected waste is recycled, composted, or sent to the landfill in fixed shares. Recycled material
    and compost leave for industry and farms. The landfill accepts waste until it nears its capacity, and a table
    function then turns the residual waste to open dumping. Waste is conserved, so the landfill fills up to its
    capacity and dumping takes over. The structure follows the solid waste management case of Bala, Arshad and Noh
    (2017), with illustrative parameters in tonnes and years for a city of a million people.

    Args:
        population:
            The initial population.

        growth_rate:
            The yearly growth rate of the population.

        per_capita:
            The waste generated per person, in tonnes per year.

        recycling_share:
            The share of the collected waste that is recycled.

        composting_share:
            The share of the collected waste that is composted.

        landfill_capacity:
            The capacity of the landfill in tonnes.

    Returns:
        system:
            The model, with the population `city.population`, the waste stocks `city.uncollected`, `city.recycled`,
            `city.composted`, `city.landfill`, and `city.dumped` in tonnes, the capacity `city.collection_capacity`,
            and the flows `city.population_growth`, `city.waste_generation`, `city.recycling`, `city.composting`,
            `city.landfilling`, `city.dumping`, `city.recycled_sales`, and `city.compost_sales` per year, over years.
            The observables of the model are `city.generation`, `city.collection`, and `city.acceptance`.

    Examples:
        >>> from skcomplex.models.engineering.processes import solid_waste
        >>> solid_waste(landfill_capacity=1e6).elements[0].landfill_capacity
        1000000.0
    """
    city = WasteSystemElement(
        'city',
        population=population,
        collection_capacity=INITIAL_COLLECTION_SHARE * per_capita * population,
        growth_rate=growth_rate,
        per_capita=per_capita,
        recycling_share=recycling_share,
        composting_share=composting_share,
        landfill_capacity=landfill_capacity,
    )
    return System(elements=[city])


@register_model(
    field='engineering/processes',
    tags=('humanitarian logistics', 'supply chain', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('need', _need, element='relief'),
        ObservableSpec('shipments', _shipments, element='relief'),
    ],
)
def food_relief(
    affected: float = 1e5,
    *,
    ration: float = 5e-4,
    onset: float = 10.0,
    shipping_delay: float = 14.0,
    cover: float = 7.0,
    stock_time: float = 7.0,
) -> System:
    """Return a food relief system, a pipeline that ships food to a warehouse and distributes it to people in need.

    A disaster at `onset` leaves `affected` people who need a daily ration. The relief agency ships food at the need
    plus corrections that close the gaps between the warehouse stock and its desired cover of the need, and between
    the food in transit and the need over the shipping delay. Shipped food enters the pipeline and arrives at the
    warehouse through a third-order material delay. The warehouse distributes the need while its stock lasts. Food is
    conserved, so almost nothing reaches the people in the first days after the disaster, and the gap closes as the
    pipeline fills. The structure follows the relief food example of Bala, Arshad and Noh (2017, Section 4.11), with
    illustrative parameters in tonnes and days.

    Args:
        affected:
            The number of people in need after the disaster.

        ration:
            The daily ration in tonnes per person.

        onset:
            The day of the disaster.

        shipping_delay:
            The average time from shipping to arrival, in days.

        cover:
            The desired warehouse stock in days of need.

        stock_time:
            The time to correct the warehouse stock, in days.

    Returns:
        system:
            The model, with the food stocks `relief.in_transit` and `relief.warehouse` in tonnes, and the flows
            `relief.shipping`, `relief.arrivals`, and `relief.distribution` in tonnes per day, over days. The
            observables of the model are `relief.need` and `relief.shipments`.

    Examples:
        >>> from skcomplex.models.engineering.processes import food_relief
        >>> food_relief(shipping_delay=7.0).elements[0].shipping_delay
        7.0
    """
    relief = ReliefPipelineElement(
        'relief',
        affected=affected,
        ration=ration,
        onset=onset,
        shipping_delay=shipping_delay,
        cover=cover,
        stock_time=stock_time,
    )
    return System(elements=[relief])


@register_model(
    field='engineering/processes',
    tags=('water resources', 'agriculture', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 730.0, 'dt': 1.0},
    observables=lambda system: [
        ObservableSpec('demand', _demand, element='scheme'),
        ObservableSpec('release', _release, element='scheme'),
        ObservableSpec('pumping', _pumping, element='scheme'),
    ],
)
def irrigation(
    *,
    river_inflow: float = 0.5,
    seasonality: float = 0.9,
    reservoir_capacity: float = 80.0,
    potential_et: float = 0.4,
    canal_loss: float = 0.2,
    pump_capacity: float = 0.2,
) -> System:
    """Return an irrigation system, a reservoir and an aquifer that keep the root zone of the crops supplied.

    A river fills the reservoir with a yearly cycle, and the reservoir loses water to evaporation and spills above its
    capacity. The irrigation demand is the potential crop evapotranspiration plus the refill of the root zone towards
    a target fraction of its field capacity. The reservoir releases the demand grossed up for the canal losses, as far
    as its stock allows, and the losses seep into the aquifer. When the release falls short, the farmers pump the
    unmet demand from the aquifer up to the pump capacity. The crops draw water from the root zone at the potential
    rate scaled by a water-stress table, and the root zone percolates to the aquifer, which drains as baseflow. Water
    is conserved, so in the dry season the reservoir falls, the pumping rises, and the aquifer is drawn down. The
    structure follows the irrigation example of Bala, Arshad and Noh (2017, Section 4.11), with illustrative
    parameters in millions of cubic metres and days.

    Args:
        river_inflow:
            The mean river inflow to the reservoir, per day.

        seasonality:
            The relative amplitude of the yearly inflow cycle.

        reservoir_capacity:
            The capacity of the reservoir.

        potential_et:
            The crop water requirement, the potential evapotranspiration of the irrigated area per day.

        canal_loss:
            The fraction of the release lost from the canals to the aquifer.

        pump_capacity:
            The largest groundwater pumping per day.

    Returns:
        system:
            The model, with the water stocks `scheme.reservoir`, `scheme.soil`, and `scheme.groundwater`, and the
            flows `scheme.inflow`, `scheme.evaporation`, `scheme.spill`, `scheme.irrigation`, `scheme.seepage`,
            `scheme.groundwater_pumping`, `scheme.crop_water`, `scheme.percolation`, and `scheme.baseflow` per day,
            over days. The observables of the model are `scheme.demand`, `scheme.release`, and `scheme.pumping`.

    Examples:
        >>> from skcomplex.models.engineering.processes import irrigation
        >>> irrigation(pump_capacity=0.0).elements[0].pump_capacity
        0.0
    """
    scheme = IrrigationSchemeElement(
        'scheme',
        river_inflow=river_inflow,
        seasonality=seasonality,
        reservoir_capacity=reservoir_capacity,
        potential_et=potential_et,
        canal_loss=canal_loss,
        pump_capacity=pump_capacity,
    )
    return System(elements=[scheme])
