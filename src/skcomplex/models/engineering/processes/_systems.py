"""Process systems: stocks of grain, waste, food, and water that engineered systems move between holders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, Expression, parameter, state
from ....core.model import (
    Auxiliary,
    Element,
    Law,
    Rate,
    Sink,
    Source,
    Symbols,
    System,
    Transfer,
    lookup,
    step,
    where,
)
from ....core.runtime import register_model
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


def _rice_mill_laws(s: Symbols) -> list[Law]:
    """Return the procurement, milling, and sales of a rice mill, and the adjustment of its milling capacity."""
    harvest = _season(s.harvest, s.seasonality, s.t, MONTHS_PER_YEAR)
    procurement = Auxiliary('procurement', delay3(harvest, s.procurement_delay))
    indicated = Auxiliary('indicated_milling', s.paddy / s.milling_time)
    milling = Auxiliary('milling', _smaller(indicated.symbol, s.capacity))
    losses = 1 - s.outturn - s.husk_share - s.bran_share
    return [
        procurement,
        indicated,
        milling,
        Transfer(Source('farms'), s.paddy, procurement.symbol, label='paddy_purchases'),
        Transfer(s.paddy, s.rice, s.outturn * milling.symbol, label='milled_rice'),
        Transfer(s.paddy, s.husk, s.husk_share * milling.symbol, label='husk_output'),
        Transfer(s.paddy, s.bran, s.bran_share * milling.symbol, label='bran_output'),
        Transfer(s.paddy, Sink('losses'), losses * milling.symbol, label='milling_losses'),
        Transfer(s.rice, Sink('market'), s.rice / s.dispatch_time, label='rice_sales'),
        Transfer(s.husk, Sink('fuel'), s.husk / s.by_product_time, label='husk_sales'),
        Transfer(s.bran, Sink('feed'), s.bran / s.by_product_time, label='bran_sales'),
        Rate(s.capacity, (smooth(indicated.symbol, s.perception_time) - s.capacity) / s.expansion_time),
    ]


def _waste_laws(s: Symbols) -> list[Law]:
    """Return the generation, collection, treatment, and disposal of the solid waste of a growing city."""
    generation = Auxiliary('generation', s.per_capita * s.population)
    collection = Auxiliary('collection', _smaller(s.uncollected / s.collection_time, s.collection_capacity))
    acceptance = Auxiliary('acceptance', lookup(s.landfill / s.landfill_capacity, LANDFILL_ACCEPTANCE))
    residual = (1 - s.recycling_share - s.composting_share) * collection.symbol
    return [
        generation,
        collection,
        acceptance,
        Transfer(Source('births'), s.population, s.growth_rate * s.population, label='population_growth'),
        Transfer(Source('households'), s.uncollected, generation.symbol, label='waste_generation'),
        Transfer(s.uncollected, s.recycled, s.recycling_share * collection.symbol, label='recycling'),
        Transfer(s.uncollected, s.composted, s.composting_share * collection.symbol, label='composting'),
        Transfer(s.uncollected, s.landfill, acceptance.symbol * residual, label='landfilling'),
        Transfer(s.uncollected, s.dumped, (1 - acceptance.symbol) * residual, label='dumping'),
        Transfer(s.recycled, Sink('industry'), s.recycled / s.recycling_time, label='recycled_sales'),
        Transfer(s.composted, Sink('farms'), s.composted / s.composting_time, label='compost_sales'),
        Rate(s.collection_capacity, (generation.symbol - s.collection_capacity) / s.capacity_time),
    ]


def _relief_laws(s: Symbols) -> list[Law]:
    """Return the shipments, arrivals, and distribution of relief food after a disaster."""
    need = Auxiliary('need', s.ration * s.affected * step(s.t, s.onset))
    stock_gap = (s.cover * need.symbol - s.warehouse) / s.stock_time
    pipeline_gap = (s.shipping_delay * need.symbol - s.in_transit) / s.pipeline_time
    shipments = Auxiliary('shipments', _positive(need.symbol + stock_gap + pipeline_gap))
    return [
        need,
        shipments,
        Transfer(Source('donors'), s.in_transit, shipments.symbol, label='shipping'),
        Transfer(s.in_transit, s.warehouse, delay3(shipments.symbol, s.shipping_delay), label='arrivals'),
        Transfer(
            s.warehouse,
            Sink('beneficiaries'),
            _smaller(need.symbol, s.warehouse / s.distribution_time),
            label='distribution',
        ),
    ]


def _irrigation_laws(s: Symbols) -> list[Law]:
    """Return the water balance of a reservoir, the root zone of the irrigated crops, and the aquifer below them."""
    demand = Auxiliary('demand', _positive(s.potential_et + (s.target * s.field_capacity - s.soil) / s.refill_time))
    release = Auxiliary('release', _smaller(demand.symbol / (1 - s.canal_loss), s.reservoir / s.release_time))
    unmet = demand.symbol - (1 - s.canal_loss) * release.symbol
    pumping = Auxiliary('pumping', _smaller(_positive(unmet), _smaller(s.pump_capacity, s.groundwater / s.pump_time)))
    crop_water = s.potential_et * lookup(s.soil / s.field_capacity, WATER_STRESS)
    spill = _positive(s.reservoir - s.reservoir_capacity) / s.spill_time
    return [
        demand,
        release,
        pumping,
        Transfer(
            Source('river'),
            s.reservoir,
            _season(s.river_inflow, s.seasonality, s.t, DAYS_PER_YEAR),
            label='inflow',
        ),
        Transfer(s.reservoir, Sink('atmosphere'), s.evaporation * s.reservoir, label='evaporation'),
        Transfer(s.reservoir, Sink('downstream'), spill, label='spill'),
        Transfer(s.reservoir, s.soil, (1 - s.canal_loss) * release.symbol, label='irrigation'),
        Transfer(s.reservoir, s.groundwater, s.canal_loss * release.symbol, label='seepage'),
        Transfer(s.groundwater, s.soil, pumping.symbol, label='groundwater_pumping'),
        Transfer(s.soil, Sink('atmosphere'), crop_water, label='crop_water'),
        Transfer(s.soil, s.groundwater, s.percolation * s.soil, label='percolation'),
        Transfer(s.groundwater, Sink('downstream'), s.groundwater / s.baseflow_time, label='baseflow'),
    ]


RiceMill = Element.define(
    'RiceMill',
    states={
        'paddy': state(default=2000.0, domain=Domain.NONNEGATIVE, quantity='grain'),
        'rice': state(default=1300.0, domain=Domain.NONNEGATIVE, quantity='grain'),
        'husk': state(default=400.0, domain=Domain.NONNEGATIVE, quantity='grain'),
        'bran': state(default=160.0, domain=Domain.NONNEGATIVE, quantity='grain'),
        'capacity': state(default=2000.0, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'harvest': parameter(default=2000.0, domain=Domain.NONNEGATIVE),
        'seasonality': parameter(default=0.8, domain=Domain.NONNEGATIVE),
        'procurement_delay': parameter(default=1.0, domain=Domain.POSITIVE),
        'milling_time': parameter(default=1.0, domain=Domain.POSITIVE),
        'outturn': parameter(default=0.65, domain=Domain.NONNEGATIVE),
        'husk_share': parameter(default=HUSK_SHARE, domain=Domain.NONNEGATIVE),
        'bran_share': parameter(default=BRAN_SHARE, domain=Domain.NONNEGATIVE),
        'dispatch_time': parameter(default=1.0, domain=Domain.POSITIVE),
        'by_product_time': parameter(default=1.0, domain=Domain.POSITIVE),
        'perception_time': parameter(default=6.0, domain=Domain.POSITIVE),
        'expansion_time': parameter(default=12.0, domain=Domain.POSITIVE),
    },
    laws=_rice_mill_laws,
)

WasteSystem = Element.define(
    'WasteSystem',
    states={
        'population': state(default=1e6, domain=Domain.NONNEGATIVE, quantity='people'),
        'uncollected': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='waste'),
        'recycled': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='waste'),
        'composted': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='waste'),
        'landfill': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='waste'),
        'dumped': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='waste'),
        'collection_capacity': state(default=1.5e5, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'growth_rate': parameter(default=0.02, domain=Domain.REAL),
        'per_capita': parameter(default=0.2, domain=Domain.NONNEGATIVE),
        'collection_time': parameter(default=0.05, domain=Domain.POSITIVE),
        'recycling_share': parameter(default=0.2, domain=Domain.NONNEGATIVE),
        'composting_share': parameter(default=0.3, domain=Domain.NONNEGATIVE),
        'landfill_capacity': parameter(default=2e6, domain=Domain.POSITIVE),
        'recycling_time': parameter(default=0.25, domain=Domain.POSITIVE),
        'composting_time': parameter(default=0.5, domain=Domain.POSITIVE),
        'capacity_time': parameter(default=3.0, domain=Domain.POSITIVE),
    },
    laws=_waste_laws,
)

ReliefPipeline = Element.define(
    'ReliefPipeline',
    states={
        'in_transit': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='food'),
        'warehouse': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='food'),
    },
    parameters={
        'affected': parameter(default=1e5, domain=Domain.NONNEGATIVE),
        'ration': parameter(default=5e-4, domain=Domain.NONNEGATIVE),
        'onset': parameter(default=10.0, domain=Domain.NONNEGATIVE),
        'shipping_delay': parameter(default=14.0, domain=Domain.POSITIVE),
        'cover': parameter(default=7.0, domain=Domain.NONNEGATIVE),
        'stock_time': parameter(default=7.0, domain=Domain.POSITIVE),
        'pipeline_time': parameter(default=14.0, domain=Domain.POSITIVE),
        'distribution_time': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=_relief_laws,
)

IrrigationScheme = Element.define(
    'IrrigationScheme',
    states={
        'reservoir': state(default=50.0, domain=Domain.NONNEGATIVE, quantity='water'),
        'soil': state(default=12.0, domain=Domain.NONNEGATIVE, quantity='water'),
        'groundwater': state(default=100.0, domain=Domain.NONNEGATIVE, quantity='water'),
    },
    parameters={
        'river_inflow': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'seasonality': parameter(default=0.9, domain=Domain.NONNEGATIVE),
        'reservoir_capacity': parameter(default=80.0, domain=Domain.POSITIVE),
        'evaporation': parameter(default=0.001, domain=Domain.NONNEGATIVE),
        'spill_time': parameter(default=1.0, domain=Domain.POSITIVE),
        'release_time': parameter(default=10.0, domain=Domain.POSITIVE),
        'canal_loss': parameter(default=0.2, domain=Domain.NONNEGATIVE),
        'potential_et': parameter(default=0.4, domain=Domain.NONNEGATIVE),
        'field_capacity': parameter(default=20.0, domain=Domain.POSITIVE),
        'target': parameter(default=0.7, domain=Domain.NONNEGATIVE),
        'refill_time': parameter(default=5.0, domain=Domain.POSITIVE),
        'percolation': parameter(default=0.005, domain=Domain.NONNEGATIVE),
        'pump_capacity': parameter(default=0.2, domain=Domain.NONNEGATIVE),
        'pump_time': parameter(default=10.0, domain=Domain.POSITIVE),
        'baseflow_time': parameter(default=500.0, domain=Domain.POSITIVE),
    },
    laws=_irrigation_laws,
)


@register_model(
    field='engineering/processes',
    tags=('agriculture', 'milling', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.25},
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
            capacity `mill.capacity`, the auxiliaries `mill.procurement`, `mill.indicated_milling`, and `mill.milling`,
            and the flows `mill.paddy_purchases`, `mill.milled_rice`, `mill.husk_output`, `mill.bran_output`,
            `mill.milling_losses`, `mill.rice_sales`, `mill.husk_sales`, and `mill.bran_sales` in tonnes per month,
            over months.

    Examples:
        >>> from skcomplex.models.engineering.processes import rice_milling
        >>> rice_milling(capacity=1500.0).elements[0].capacity
        1500.0
    """
    mill = RiceMill(
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
    return System(components=[mill])


@register_model(
    field='engineering/processes',
    tags=('waste management', 'urban', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 40.0, 'dt': 0.1},
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
            the auxiliaries `city.generation`, `city.collection`, and `city.acceptance`, and the flows
            `city.population_growth`, `city.waste_generation`, `city.recycling`, `city.composting`,
            `city.landfilling`, `city.dumping`, `city.recycled_sales`, and `city.compost_sales` per year, over years.

    Examples:
        >>> from skcomplex.models.engineering.processes import solid_waste
        >>> solid_waste(landfill_capacity=1e6).elements[0].landfill_capacity
        1000000.0
    """
    city = WasteSystem(
        'city',
        population=population,
        collection_capacity=INITIAL_COLLECTION_SHARE * per_capita * population,
        growth_rate=growth_rate,
        per_capita=per_capita,
        recycling_share=recycling_share,
        composting_share=composting_share,
        landfill_capacity=landfill_capacity,
    )
    return System(components=[city])


@register_model(
    field='engineering/processes',
    tags=('humanitarian logistics', 'supply chain', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.5},
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
            The model, with the food stocks `relief.in_transit` and `relief.warehouse` in tonnes, the auxiliaries
            `relief.need` and `relief.shipments`, and the flows `relief.shipping`, `relief.arrivals`, and
            `relief.distribution` in tonnes per day, over days.

    Examples:
        >>> from skcomplex.models.engineering.processes import food_relief
        >>> food_relief(shipping_delay=7.0).elements[0].shipping_delay
        7.0
    """
    relief = ReliefPipeline(
        'relief',
        affected=affected,
        ration=ration,
        onset=onset,
        shipping_delay=shipping_delay,
        cover=cover,
        stock_time=stock_time,
    )
    return System(components=[relief])


@register_model(
    field='engineering/processes',
    tags=('water resources', 'agriculture', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 730.0, 'dt': 1.0},
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
            The model, with the water stocks `scheme.reservoir`, `scheme.soil`, and `scheme.groundwater`, the
            auxiliaries `scheme.demand`, `scheme.release`, and `scheme.pumping`, and the flows `scheme.inflow`,
            `scheme.evaporation`, `scheme.spill`, `scheme.irrigation`, `scheme.seepage`, `scheme.groundwater_pumping`,
            `scheme.crop_water`, `scheme.percolation`, and `scheme.baseflow` per day, over days.

    Examples:
        >>> from skcomplex.models.engineering.processes import irrigation
        >>> irrigation(pump_capacity=0.0).elements[0].pump_capacity
        0.0
    """
    scheme = IrrigationScheme(
        'scheme',
        river_inflow=river_inflow,
        seasonality=seasonality,
        reservoir_capacity=reservoir_capacity,
        potential_et=potential_et,
        canal_loss=canal_loss,
        pump_capacity=pump_capacity,
    )
    return System(components=[scheme])
