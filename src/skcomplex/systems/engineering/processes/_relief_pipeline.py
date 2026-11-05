"""Relief supply line that ships food through a delayed pipeline to a warehouse and distributes it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import clip, delay3, minimum, step
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ReliefPipelineElement(Element):
    """Relief supply line that ships food through a delayed pipeline to a warehouse and distributes it."""

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
        """Return the need of relief food, which starts at the onset of the disaster."""
        return s.ration * s.affected * step(s.t, s.onset)

    @staticmethod
    def _shipments(s: SimpleNamespace) -> Expression:
        """Return the shipments that meet the need and close the gaps of the stock and of the pipeline."""
        need = ReliefPipelineElement._need(s)
        stock_gap = (s.cover * need - s.warehouse) / s.stock_time
        pipeline_gap = (s.shipping_delay * need - s.in_transit) / s.pipeline_time
        return clip(need + stock_gap + pipeline_gap, 0)

    @staticmethod
    def _relief_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shipments, arrivals, and distribution of relief food after a disaster."""
        shipments = ReliefPipelineElement._shipments(s)
        return [
            TransferLaw(Source('donors'), s.in_transit, shipments, label='shipping'),
            TransferLaw(s.in_transit, s.warehouse, delay3(shipments, s.shipping_delay), label='arrivals'),
            TransferLaw(
                s.warehouse,
                Sink('beneficiaries'),
                minimum(ReliefPipelineElement._need(s), s.warehouse / s.distribution_time),
                label='distribution',
            ),
        ]
