"""Irrigation scheme that moves water between a reservoir, the root zone of crops, and an aquifer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.laws.utils import clip, lookup, minimum
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._rice_mill import RiceMillElement

DAYS_PER_YEAR = 365.0


WATER_STRESS = ((0.0, 0.0), (0.3, 0.5), (0.6, 1.0), (1.0, 1.0))


class IrrigationSchemeElement(Element):
    """Irrigation scheme that moves water between a reservoir, the root zone of crops, and an aquifer."""

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
        """Return the water demand of the crops, their evapotranspiration and the refill of the root zone."""
        return clip(s.potential_et + (s.target * s.field_capacity - s.soil) / s.refill_time, 0)

    @staticmethod
    def _reservoir_release(s: SimpleNamespace) -> Expression:
        """Return the release of the reservoir, the demand before the canal losses, limited by the reservoir."""
        return minimum(IrrigationSchemeElement._demand(s) / (1 - s.canal_loss), s.reservoir / s.release_time)

    @staticmethod
    def _pumping(s: SimpleNamespace) -> Expression:
        """Return the pumping of groundwater, the demand the release leaves, limited by the pumps and the aquifer."""
        unmet = IrrigationSchemeElement._demand(s) - (1 - s.canal_loss) * IrrigationSchemeElement._reservoir_release(s)
        return minimum(clip(unmet, 0), minimum(s.pump_capacity, s.groundwater / s.pump_time))

    @staticmethod
    def _irrigation_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the water balance of a reservoir, the root zone of the irrigated crops, and the aquifer below them."""
        release = IrrigationSchemeElement._reservoir_release(s)
        crop_water = s.potential_et * lookup(s.soil / s.field_capacity, WATER_STRESS)
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
