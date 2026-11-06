"""Process systems: stocks of grain, waste, food, and water that engineered systems move between holders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._irrigation_scheme import IrrigationSchemeElement
from ._relief_pipeline import ReliefPipelineElement
from ._rice_mill import BRAN_SHARE, HUSK_SHARE, RiceMillElement
from ._waste_system import WasteSystemElement

INITIAL_COLLECTION_SHARE = 0.75


@register_system(
    field='engineering/processes',
    tags=('agriculture', 'milling', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.25},
    observables=lambda system: [
        ObservableSpec('procurement', RiceMillElement._procurement, element='mill'),
        ObservableSpec('indicated_milling', RiceMillElement._indicated_milling, element='mill'),
        ObservableSpec('milling', RiceMillElement._milling, element='mill'),
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
            The system, with the grain stocks `mill.paddy`, `mill.rice`, `mill.husk`, and `mill.bran` in tonnes, the
            capacity `mill.capacity`, and the flows `mill.paddy_purchases`, `mill.milled_rice`, `mill.husk_output`,
            `mill.bran_output`, `mill.milling_losses`, `mill.rice_sales`, `mill.husk_sales`, and `mill.bran_sales` in
            tonnes per month, over months. The observables of the system are `mill.procurement`,
            `mill.indicated_milling`, and `mill.milling`.

    Examples:
        >>> from skcomplex.systems.engineering.processes import rice_milling
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


@register_system(
    field='engineering/processes',
    tags=('waste management', 'urban', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 40.0, 'dt': 0.1},
    observables=lambda system: [
        ObservableSpec('generation', WasteSystemElement._generation, element='city'),
        ObservableSpec('collection', WasteSystemElement._collection, element='city'),
        ObservableSpec('acceptance', WasteSystemElement._acceptance, element='city'),
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
            The system, with the population `city.population`, the waste stocks `city.uncollected`, `city.recycled`,
            `city.composted`, `city.landfill`, and `city.dumped` in tonnes, the capacity `city.collection_capacity`,
            and the flows `city.population_growth`, `city.waste_generation`, `city.recycling`, `city.composting`,
            `city.landfilling`, `city.dumping`, `city.recycled_sales`, and `city.compost_sales` per year, over years.
            The observables of the system are `city.generation`, `city.collection`, and `city.acceptance`.

    Examples:
        >>> from skcomplex.systems.engineering.processes import solid_waste
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


@register_system(
    field='engineering/processes',
    tags=('humanitarian logistics', 'supply chain', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('need', ReliefPipelineElement._need, element='relief'),
        ObservableSpec('shipments', ReliefPipelineElement._shipments, element='relief'),
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
            The system, with the food stocks `relief.in_transit` and `relief.warehouse` in tonnes, and the flows
            `relief.shipping`, `relief.arrivals`, and `relief.distribution` in tonnes per day, over days. The
            observables of the system are `relief.need` and `relief.shipments`.

    Examples:
        >>> from skcomplex.systems.engineering.processes import food_relief
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


@register_system(
    field='engineering/processes',
    tags=('water resources', 'agriculture', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 730.0, 'dt': 1.0},
    observables=lambda system: [
        ObservableSpec('demand', IrrigationSchemeElement._demand, element='scheme'),
        ObservableSpec('release', IrrigationSchemeElement._reservoir_release, element='scheme'),
        ObservableSpec('pumping', IrrigationSchemeElement._pumping, element='scheme'),
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
            The system, with the water stocks `scheme.reservoir`, `scheme.soil`, and `scheme.groundwater`, and the
            flows `scheme.inflow`, `scheme.evaporation`, `scheme.spill`, `scheme.irrigation`, `scheme.seepage`,
            `scheme.groundwater_pumping`, `scheme.crop_water`, `scheme.percolation`, and `scheme.baseflow` per day,
            over days. The observables of the system are `scheme.demand`, `scheme.release`, and `scheme.pumping`.

    Examples:
        >>> from skcomplex.systems.engineering.processes import irrigation
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
