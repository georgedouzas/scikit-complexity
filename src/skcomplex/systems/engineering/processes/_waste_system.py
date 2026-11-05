"""Solid waste system of a growing city that collects, recycles, composts, and landfills its waste."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import lookup, minimum
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

LANDFILL_ACCEPTANCE = ((0.0, 1.0), (0.9, 1.0), (1.0, 0.0))


class WasteSystemElement(Element):
    """Solid waste system of a growing city that collects, recycles, composts, and landfills its waste."""

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
        """Return the flows of waste through the city and the adjustment of its collection capacity."""
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
        return lookup(s.landfill / s.landfill_capacity, LANDFILL_ACCEPTANCE)

    @staticmethod
    def _waste_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the generation, collection, treatment, and disposal of the solid waste of a growing city."""
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
