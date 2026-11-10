"""Process systems: stocks of grain, waste, food, and water.

Engineered systems move the stocks between holders.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import clip, distributed_delay, minimum, series, step
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

MONTHS_PER_YEAR = 12.0
HUSK_SHARE = 0.2
BRAN_SHARE = 0.08
DAYS_PER_YEAR = 365.0
WATER_STRESS = ((0.0, 0.3, 0.6, 1.0), (0.0, 0.5, 1.0, 1.0))
LANDFILL_ACCEPTANCE = ((0.0, 0.9, 1.0), (1.0, 1.0, 0.0))
INITIAL_COLLECTION_SHARE = 0.75


class ReliefPipelineElement(Element):
    """Relief supply line that ships food to a warehouse and distributes it.

    The food is shipped through a delayed pipeline.
    """

    law_classes = (TransferLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        in_transit: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='food')] = 0.0,
        warehouse: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='food')] = 0.0,
        affected: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100000.0,
        ration: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0005,
        onset: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        shipping_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 14.0,
        cover: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 7.0,
        stock_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 7.0,
        pipeline_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 14.0,
        distribution_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.in_transit = in_transit
        self.warehouse = warehouse
        self.affected = affected
        self.ration = ration
        self.onset = onset
        self.shipping_delay = shipping_delay
        self.cover = cover
        self.stock_time = stock_time
        self.pipeline_time = pipeline_time
        self.distribution_time = distribution_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of food through the supply line."""
        return ReliefPipelineElement._relief_laws(s)

    @staticmethod
    def _need(s: SimpleNamespace) -> Expression:
        """Return the need of relief food.

        It starts at the onset of the disaster.
        """
        return s.ration * s.affected * step(s.t, s.onset)

    @staticmethod
    def _shipments(s: SimpleNamespace) -> Expression:
        """Return the shipments that meet the need and close the gaps.

        The gaps are those of the stock and of the pipeline.
        """
        need = ReliefPipelineElement._need(s)
        stock_gap = (s.cover * need - s.warehouse) / s.stock_time
        pipeline_gap = (s.shipping_delay * need - s.in_transit) / s.pipeline_time
        return clip(need + stock_gap + pipeline_gap, 0)

    @staticmethod
    def _relief_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shipments, arrivals, and distribution of relief food.

        They follow a disaster.
        """
        shipments = ReliefPipelineElement._shipments(s)
        return [
            TransferLaw(Source('donors'), s.in_transit, shipments, label='shipping'),
            TransferLaw(s.in_transit, s.warehouse, distributed_delay(shipments, s.shipping_delay, 3), label='arrivals'),
            TransferLaw(
                s.warehouse,
                Sink('beneficiaries'),
                minimum(ReliefPipelineElement._need(s), s.warehouse / s.distribution_time),
                label='distribution',
            ),
        ]


class RiceMillElement(Element):
    """Rice mill that buys seasonal paddy, mills it, and adjusts its capacity.

    The paddy is milled into rice, husk, and bran.
    """

    law_classes = (TransferLaw, RateLaw)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        paddy: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='grain')] = 2000.0,
        rice: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='grain')] = 1300.0,
        husk: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='grain')] = 400.0,
        bran: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='grain')] = 160.0,
        capacity: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        harvest: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        seasonality: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        procurement_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        milling_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        outturn: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.65,
        husk_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = HUSK_SHARE,
        bran_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = BRAN_SHARE,
        dispatch_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        by_product_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        perception_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 6.0,
        expansion_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.paddy = paddy
        self.rice = rice
        self.husk = husk
        self.bran = bran
        self.capacity = capacity
        self.harvest = harvest
        self.seasonality = seasonality
        self.procurement_delay = procurement_delay
        self.milling_time = milling_time
        self.outturn = outturn
        self.husk_share = husk_share
        self.bran_share = bran_share
        self.dispatch_time = dispatch_time
        self.by_product_time = by_product_time
        self.perception_time = perception_time
        self.expansion_time = expansion_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of grain through the mill and the adjustment of its capacity."""
        return RiceMillElement._rice_mill_laws(s)

    @staticmethod
    def _season(mean: Expression, amplitude: Expression, t: Expression, period: float) -> Expression:
        """Return a seasonal value that oscillates about its mean.

        It has a relative amplitude and a period.
        """
        return mean * (1 + amplitude * sin(2 * math.pi * t / period))

    @staticmethod
    def _procurement(s: SimpleNamespace) -> Expression:
        """Return the procurement of paddy.

        It is the seasonal harvest after the procurement delay.
        """
        season = RiceMillElement._season(s.harvest, s.seasonality, s.t, MONTHS_PER_YEAR)
        return distributed_delay(season, s.procurement_delay, 3)

    @staticmethod
    def _indicated_milling(s: SimpleNamespace) -> Expression:
        """Return the indicated milling, the paddy milled over the milling time."""
        return s.paddy / s.milling_time

    @staticmethod
    def _milling(s: SimpleNamespace) -> Expression:
        """Return the milling, the indicated milling limited by the capacity."""
        return minimum(RiceMillElement._indicated_milling(s), s.capacity)

    @staticmethod
    def _rice_mill_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the procurement, milling, and sales of a rice mill.

        It also returns the adjustment of its milling capacity.
        """
        milling = RiceMillElement._milling(s)
        losses = 1 - s.outturn - s.husk_share - s.bran_share
        perceived = distributed_delay(RiceMillElement._indicated_milling(s), s.perception_time, conserved=False)
        return [
            TransferLaw(Source('farms'), s.paddy, RiceMillElement._procurement(s), label='paddy_purchases'),
            TransferLaw(s.paddy, s.rice, s.outturn * milling, label='milled_rice'),
            TransferLaw(s.paddy, s.husk, s.husk_share * milling, label='husk_output'),
            TransferLaw(s.paddy, s.bran, s.bran_share * milling, label='bran_output'),
            TransferLaw(s.paddy, Sink('losses'), losses * milling, label='milling_losses'),
            TransferLaw(s.rice, Sink('market'), s.rice / s.dispatch_time, label='rice_sales'),
            TransferLaw(s.husk, Sink('fuel'), s.husk / s.by_product_time, label='husk_sales'),
            TransferLaw(s.bran, Sink('feed'), s.bran / s.by_product_time, label='bran_sales'),
            RateLaw(
                s.capacity,
                (perceived - s.capacity) / s.expansion_time,
            ),
        ]


class IrrigationSchemeElement(Element):
    """Irrigation scheme that moves water between its reservoirs.

    The water moves between a reservoir, the root zone of crops, and an aquifer.
    """

    law_classes = (TransferLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        reservoir: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='water')] = 50.0,
        soil: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='water')] = 12.0,
        groundwater: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='water')] = 100.0,
        river_inflow: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        seasonality: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        reservoir_capacity: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 80.0,
        evaporation: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        spill_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        release_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        canal_loss: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        potential_et: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        field_capacity: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        target: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        refill_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        percolation: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        pump_capacity: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        pump_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        baseflow_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 500.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.reservoir = reservoir
        self.soil = soil
        self.groundwater = groundwater
        self.river_inflow = river_inflow
        self.seasonality = seasonality
        self.reservoir_capacity = reservoir_capacity
        self.evaporation = evaporation
        self.spill_time = spill_time
        self.release_time = release_time
        self.canal_loss = canal_loss
        self.potential_et = potential_et
        self.field_capacity = field_capacity
        self.target = target
        self.refill_time = refill_time
        self.percolation = percolation
        self.pump_capacity = pump_capacity
        self.pump_time = pump_time
        self.baseflow_time = baseflow_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of water through the scheme."""
        return IrrigationSchemeElement._irrigation_laws(s)

    @staticmethod
    def _demand(s: SimpleNamespace) -> Expression:
        """Return the water demand of the crops.

        It is their evapotranspiration and the refill of the root zone.
        """
        return clip(s.potential_et + (s.target * s.field_capacity - s.soil) / s.refill_time, 0)

    @staticmethod
    def _reservoir_release(s: SimpleNamespace) -> Expression:
        """Return the release of the reservoir.

        It is the demand before the canal losses, limited by the reservoir.
        """
        return minimum(IrrigationSchemeElement._demand(s) / (1 - s.canal_loss), s.reservoir / s.release_time)

    @staticmethod
    def _pumping(s: SimpleNamespace) -> Expression:
        """Return the pumping of groundwater.

        It is the demand the release leaves, limited by the pumps and the aquifer.
        """
        unmet = IrrigationSchemeElement._demand(s) - (1 - s.canal_loss) * IrrigationSchemeElement._reservoir_release(s)
        return minimum(clip(unmet, 0), minimum(s.pump_capacity, s.groundwater / s.pump_time))

    @staticmethod
    def _irrigation_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the water balance of a reservoir, the root zone, and the aquifer.

        The root zone is that of the irrigated crops, and the aquifer is below them.
        """
        release = IrrigationSchemeElement._reservoir_release(s)
        crop_water = s.potential_et * series(s.soil / s.field_capacity, *WATER_STRESS)
        spill = clip(s.reservoir - s.reservoir_capacity, 0) / s.spill_time
        return [
            TransferLaw(
                Source('river'),
                s.reservoir,
                RiceMillElement._season(s.river_inflow, s.seasonality, s.t, DAYS_PER_YEAR),
                label='inflow',
            ),
            TransferLaw(s.reservoir, Sink('atmosphere'), s.evaporation * s.reservoir, label='evaporation'),
            TransferLaw(s.reservoir, Sink('downstream'), spill, label='spill'),
            TransferLaw(s.reservoir, s.soil, (1 - s.canal_loss) * release, label='irrigation'),
            TransferLaw(s.reservoir, s.groundwater, s.canal_loss * release, label='seepage'),
            TransferLaw(s.groundwater, s.soil, IrrigationSchemeElement._pumping(s), label='groundwater_pumping'),
            TransferLaw(s.soil, Sink('atmosphere'), crop_water, label='crop_water'),
            TransferLaw(s.soil, s.groundwater, s.percolation * s.soil, label='percolation'),
            TransferLaw(s.groundwater, Sink('downstream'), s.groundwater / s.baseflow_time, label='baseflow'),
        ]


class WasteSystemElement(Element):
    """Solid waste system of a growing city.

    It collects, recycles, composts, and landfills its waste.
    """

    law_classes = (TransferLaw, RateLaw)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        population: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 1000000.0,
        uncollected: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='waste')] = 0.0,
        recycled: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='waste')] = 0.0,
        composted: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='waste')] = 0.0,
        landfill: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='waste')] = 0.0,
        dumped: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='waste')] = 0.0,
        collection_capacity: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 150000.0,
        growth_rate: Annotated[object, ParameterSpec()] = 0.02,
        per_capita: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        collection_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        recycling_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        composting_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        landfill_capacity: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2000000.0,
        recycling_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
        composting_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        capacity_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.population = population
        self.uncollected = uncollected
        self.recycled = recycled
        self.composted = composted
        self.landfill = landfill
        self.dumped = dumped
        self.collection_capacity = collection_capacity
        self.growth_rate = growth_rate
        self.per_capita = per_capita
        self.collection_time = collection_time
        self.recycling_share = recycling_share
        self.composting_share = composting_share
        self.landfill_capacity = landfill_capacity
        self.recycling_time = recycling_time
        self.composting_time = composting_time
        self.capacity_time = capacity_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of waste through the city.

        It also returns the adjustment of its collection capacity.
        """
        return WasteSystemElement._waste_laws(s)

    @staticmethod
    def _generation(s: SimpleNamespace) -> Expression:
        """Return the generation of waste by the population."""
        return s.per_capita * s.population

    @staticmethod
    def _collection(s: SimpleNamespace) -> Expression:
        """Return the collection of waste, limited by the collection capacity."""
        return minimum(s.uncollected / s.collection_time, s.collection_capacity)

    @staticmethod
    def _acceptance(s: SimpleNamespace) -> Expression:
        """Return the share of the residual waste that the landfill accepts as it fills."""
        return series(s.landfill / s.landfill_capacity, *LANDFILL_ACCEPTANCE)

    @staticmethod
    def _waste_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the generation, collection, treatment, and disposal of solid waste.

        The waste is that of a growing city.
        """
        generation, collection, acceptance = (
            WasteSystemElement._generation(s),
            WasteSystemElement._collection(s),
            WasteSystemElement._acceptance(s),
        )
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


class RiceMillingSystem(BaseSystem):
    """A rice milling system.

    A mill buys paddy, mills it into rice and by-products, and sells them. Paddy arrives from the farms through a
    third-order procurement delay, at a rate that follows the harvest season around its mean. The mill processes its
    paddy stock in a month, as far as its milling capacity allows, and splits the milled paddy by mass into rice, the
    outturn, husk, bran, and milling losses. Rice, husk, and bran leave their inventories for the market at a fixed
    coverage. The capacity adjusts to the smoothed indicated milling rate, so in the harvest season the capacity lags
    and paddy accumulates, and in the lean season the capacity idles. Grain is conserved, every change of the paddy,
    rice, husk, and bran stocks is a flow, and over a year the milled paddy and the rice sales match the mean harvest
    and its outturn. The structure follows the rice milling case of Bala, Arshad and Noh (2017), with illustrative
    parameters in tonnes and months, since the case data are specific to their study.

    The system has the grain stocks `mill.paddy`, `mill.rice`, `mill.husk`, and `mill.bran` in tonnes, the capacity
    `mill.capacity`, and the flows `mill.paddy_purchases`, `mill.milled_rice`, `mill.husk_output`, `mill.bran_output`,
    `mill.milling_losses`, `mill.rice_sales`, `mill.husk_sales`, and `mill.bran_sales` in tonnes per month, over months.
    The observables of the system are `mill.procurement`, `mill.indicated_milling`, and `mill.milling`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import RiceMillingSystem
        >>> RiceMillingSystem(capacity=1500.0).prepare().system_.elements[0].capacity
        1500.0
    """

    field = 'engineering/processes'
    topics = ('agriculture', 'milling', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 120.0
    dt = 0.25
    observables = (
        ObservableSpec('procurement', RiceMillElement._procurement, element='mill'),
        ObservableSpec('indicated_milling', RiceMillElement._indicated_milling, element='mill'),
        ObservableSpec('milling', RiceMillElement._milling, element='mill'),
    )

    def __init__(
        self,
        *,
        harvest: float = 2000.0,
        seasonality: float = 0.8,
        procurement_delay: float = 1.0,
        outturn: float = 0.65,
        capacity: float = 2000.0,
        expansion_time: float = 12.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.harvest = harvest
        self.seasonality = seasonality
        self.procurement_delay = procurement_delay
        self.outturn = outturn
        self.capacity = capacity
        self.expansion_time = expansion_time

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        mill = RiceMillElement(
            'mill',
            paddy=self.harvest,
            rice=self.outturn * self.harvest,
            husk=HUSK_SHARE * self.harvest,
            bran=BRAN_SHARE * self.harvest,
            capacity=self.capacity,
            harvest=self.harvest,
            seasonality=self.seasonality,
            procurement_delay=self.procurement_delay,
            outturn=self.outturn,
            expansion_time=self.expansion_time,
        )
        return CompositeSystem(self.label, elements=[mill])


class SolidWasteSystem(BaseSystem):
    """A solid waste system.

    The waste of a growing city is collected, recycled, composted, and landfilled. The population grows exponentially
    and generates waste in proportion to its size. The waste waits uncollected until the collection, limited by a
    collection capacity that the city expands towards the generation with a delay, removes it. The collected waste is
    recycled, composted, or sent to the landfill in fixed shares. Recycled material and compost leave for industry and
    farms. The landfill accepts waste until it nears its capacity, and a table function then turns the residual waste to
    open dumping. Waste is conserved, so the landfill fills up to its capacity and dumping takes over. The structure
    follows the solid waste management case of Bala, Arshad and Noh (2017), with illustrative parameters in tonnes and
    years for a city of a million people.

    The system has the population `city.population`, the waste stocks `city.uncollected`, `city.recycled`,
    `city.composted`, `city.landfill`, and `city.dumped` in tonnes, the capacity `city.collection_capacity`, and the
    flows `city.population_growth`, `city.waste_generation`, `city.recycling`, `city.composting`, `city.landfilling`,
    `city.dumping`, `city.recycled_sales`, and `city.compost_sales` per year, over years. The observables of the system
    are `city.generation`, `city.collection`, and `city.acceptance`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import SolidWasteSystem
        >>> SolidWasteSystem(landfill_capacity=1e6).prepare().system_.elements[0].landfill_capacity
        1000000.0
    """

    field = 'engineering/processes'
    topics = ('waste management', 'urban', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 40.0
    dt = 0.1
    observables = (
        ObservableSpec('generation', WasteSystemElement._generation, element='city'),
        ObservableSpec('collection', WasteSystemElement._collection, element='city'),
        ObservableSpec('acceptance', WasteSystemElement._acceptance, element='city'),
    )

    def __init__(
        self,
        population: float = 1e6,
        *,
        growth_rate: float = 0.02,
        per_capita: float = 0.2,
        recycling_share: float = 0.2,
        composting_share: float = 0.3,
        landfill_capacity: float = 2e6,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.population = population
        self.growth_rate = growth_rate
        self.per_capita = per_capita
        self.recycling_share = recycling_share
        self.composting_share = composting_share
        self.landfill_capacity = landfill_capacity

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        city = WasteSystemElement(
            'city',
            population=self.population,
            collection_capacity=INITIAL_COLLECTION_SHARE * self.per_capita * self.population,
            growth_rate=self.growth_rate,
            per_capita=self.per_capita,
            recycling_share=self.recycling_share,
            composting_share=self.composting_share,
            landfill_capacity=self.landfill_capacity,
        )
        return CompositeSystem(self.label, elements=[city])


class FoodReliefSystem(BaseSystem):
    """A food relief system.

    A pipeline ships food to a warehouse and distributes it to people in need. A disaster at `onset` leaves `affected`
    people who need a daily ration. The relief agency ships food at the need plus corrections that close the gaps
    between the warehouse stock and its desired cover of the need, and between the food in transit and the need over the
    shipping delay. Shipped food enters the pipeline and arrives at the warehouse through a third-order material delay.
    The warehouse distributes the need while its stock lasts. Food is conserved, so almost nothing reaches the people in
    the first days after the disaster, and the gap closes as the pipeline fills. The structure follows the relief food
    example of Bala, Arshad and Noh (2017, Section 4.11), with illustrative parameters in tonnes and days.

    The system has the food stocks `relief.in_transit` and `relief.warehouse` in tonnes, and the flows
    `relief.shipping`, `relief.arrivals`, and `relief.distribution` in tonnes per day, over days. The observables of the
    system are `relief.need` and `relief.shipments`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import FoodReliefSystem
        >>> FoodReliefSystem(shipping_delay=7.0).prepare().system_.elements[0].shipping_delay
        7.0
    """

    field = 'engineering/processes'
    topics = ('humanitarian logistics', 'supply chain', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 120.0
    dt = 0.5
    observables = (
        ObservableSpec('need', ReliefPipelineElement._need, element='relief'),
        ObservableSpec('shipments', ReliefPipelineElement._shipments, element='relief'),
    )

    def __init__(
        self,
        affected: float = 1e5,
        *,
        ration: float = 5e-4,
        onset: float = 10.0,
        shipping_delay: float = 14.0,
        cover: float = 7.0,
        stock_time: float = 7.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.affected = affected
        self.ration = ration
        self.onset = onset
        self.shipping_delay = shipping_delay
        self.cover = cover
        self.stock_time = stock_time

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        relief = ReliefPipelineElement(
            'relief',
            affected=self.affected,
            ration=self.ration,
            onset=self.onset,
            shipping_delay=self.shipping_delay,
            cover=self.cover,
            stock_time=self.stock_time,
        )
        return CompositeSystem(self.label, elements=[relief])


class IrrigationSystem(BaseSystem):
    """An irrigation system.

    A reservoir and an aquifer keep the root zone of the crops supplied. A river fills the reservoir with a yearly
    cycle, and the reservoir loses water to evaporation and spills above its capacity. The irrigation demand is the
    potential crop evapotranspiration plus the refill of the root zone towards a target fraction of its field capacity.
    The reservoir releases the demand grossed up for the canal losses, as far as its stock allows, and the losses seep
    into the aquifer. When the release falls short, the farmers pump the unmet demand from the aquifer up to the pump
    capacity. The crops draw water from the root zone at the potential rate scaled by a water-stress table, and the root
    zone percolates to the aquifer, which drains as baseflow. Water is conserved, so in the dry season the reservoir
    falls, the pumping rises, and the aquifer is drawn down. The structure follows the irrigation example of Bala,
    Arshad and Noh (2017, Section 4.11), with illustrative parameters in millions of cubic metres and days.

    The system has the water stocks `scheme.reservoir`, `scheme.soil`, and `scheme.groundwater`, and the flows
    `scheme.inflow`, `scheme.evaporation`, `scheme.spill`, `scheme.irrigation`, `scheme.seepage`,
    `scheme.groundwater_pumping`, `scheme.crop_water`, `scheme.percolation`, and `scheme.baseflow` per day, over days.
    The observables of the system are `scheme.demand`, `scheme.release`, and `scheme.pumping`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import IrrigationSystem
        >>> IrrigationSystem(pump_capacity=0.0).prepare().system_.elements[0].pump_capacity
        0.0
    """

    field = 'engineering/processes'
    topics = ('water resources', 'agriculture', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 730.0
    dt = 1.0
    observables = (
        ObservableSpec('demand', IrrigationSchemeElement._demand, element='scheme'),
        ObservableSpec('release', IrrigationSchemeElement._reservoir_release, element='scheme'),
        ObservableSpec('pumping', IrrigationSchemeElement._pumping, element='scheme'),
    )

    def __init__(
        self,
        *,
        river_inflow: float = 0.5,
        seasonality: float = 0.9,
        reservoir_capacity: float = 80.0,
        potential_et: float = 0.4,
        canal_loss: float = 0.2,
        pump_capacity: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.river_inflow = river_inflow
        self.seasonality = seasonality
        self.reservoir_capacity = reservoir_capacity
        self.potential_et = potential_et
        self.canal_loss = canal_loss
        self.pump_capacity = pump_capacity

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        scheme = IrrigationSchemeElement(
            'scheme',
            river_inflow=self.river_inflow,
            seasonality=self.seasonality,
            reservoir_capacity=self.reservoir_capacity,
            potential_et=self.potential_et,
            canal_loss=self.canal_loss,
            pump_capacity=self.pump_capacity,
        )
        return CompositeSystem(self.label, elements=[scheme])
