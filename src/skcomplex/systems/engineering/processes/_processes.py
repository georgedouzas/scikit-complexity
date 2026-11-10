"""Process systems: stocks of material that flows between stages.

The material flows in, out, and between stages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import step, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import StockFlowSystem
from ...general.dynamics.types import FlowTable

INFLOW_START = 5.0


class StageElement(Element):
    """Stage of a process, a stock of material `m` with a constant feed."""

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        m: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        feed: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.m = m
        self.feed = feed

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the constant feed of the stage."""
        return [RateLaw(s.m, s.feed)]


class BathtubSystem(StockFlowSystem):
    """A bathtub, one stock of water with an inflow and an outflow.

    The inflow is from the faucet and the outflow is through the drain. The drain empties the tub at `outflow` gallons
    per minute while there is water, and the faucet opens to `inflow` gallons per minute after five minutes. With equal
    flows the level falls for five minutes and then stays constant. The system has the water `tub.water` in gallons and
    the flows `tub.faucet` and `tub.drain` in gallons per minute, over minutes.

    Args:
        water:
            The initial water in the tub, in gallons.

        inflow:
            The inflow after five minutes, in gallons per minute.

        outflow:
            The outflow, in gallons per minute.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import BathtubSystem
        >>> BathtubSystem(water=20.0).prepare().system_.elements[0].get_value('water')
        20.0
    """

    field = 'engineering/processes'
    topics = ('stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 10.0
    dt = 0.5

    def __init__(
        self,
        water: float = 50.0,
        *,
        inflow: float = 5.0,
        outflow: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.water = water
        self.inflow = inflow
        self.outflow = outflow

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks."""
        return 'tub'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the stocks and the parameters with their specs."""
        return {
            'water': DeclaredVariableSpec(self.water, StateSpec(domain=RealSet.closed_open(0, oo), quantity='water')),
            'inflow': DeclaredVariableSpec(self.inflow, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'outflow': DeclaredVariableSpec(self.outflow, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_flows(self) -> FlowTable:
        """Return the faucet and drain flows of the tub."""
        return {
            'faucet': (None, 'water', lambda s: s.source.inflow * step(s.t, INFLOW_START)),
            'drain': ('water', None, lambda s: where(s.source.water > 0, s.source.outflow, 0)),
        }


class CascadeClassifierSystem(BaseSystem):
    """A cascade classifier, a column of stages that separates particles.

    An air flow separates them. The particles separated are of a narrow size class. Material leaves every stage at unit
    rate, a fraction `k` upwards and `1 - k` downwards. The top stage releases into the fine product and the bottom
    stage into the coarse product. Material of one size class is fed into one stage at a constant rate. At steady state
    the share of the feed that ends in the fine product is
    `F_f = (1 - ((1 - k) / k)^(z + 1 - i)) / (1 - ((1 - k) / k)^(z + 1))` for `z` stages and feeding at stage `i`
    counted from the top, Barsky's fractional extraction.

    The system has the material `stage_<i>.m` of each stage from the top, and the accumulated products `fine.m` and
    `coarse.m`.

    Args:
        n_stages:
            The number of stages `z`.

        k:
            The distribution coefficient, the fraction of material moving up from a stage.

        feed_stage:
            The stage that receives the feed, counted from 1 at the top, or `None` for the middle stage.

        feed:
            The feed rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import CascadeClassifierSystem
        >>> len(CascadeClassifierSystem(n_stages=4).prepare().system_.interactions)
        8
    """

    field = 'engineering/processes'
    topics = ('separation', 'cascade', 'stocks and flows', 'ode')
    references = ('Barsky 2020',)
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        n_stages: int = 10,
        *,
        k: float = 0.6,
        feed_stage: int | None = None,
        feed: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_stages = n_stages
        self.k = k
        self.feed_stage = feed_stage
        self.feed = feed

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        entry = (self.n_stages + 1) // 2 if self.feed_stage is None else self.feed_stage
        stages = [
            StageElement(f'stage_{index}', feed=self.feed if index == entry else 0.0)
            for index in range(1, self.n_stages + 1)
        ]
        labels = ['fine', *(stage.label for stage in stages), 'coarse']
        interactions = [
            TransferInteraction(
                f'up_{index}',
                source=labels[index],
                target=labels[index - 1],
                state='m',
                amount=lambda s: self.k * s.source.m,
            )
            for index in range(1, self.n_stages + 1)
        ] + [
            TransferInteraction(
                f'down_{index}',
                source=labels[index],
                target=labels[index + 1],
                state='m',
                amount=lambda s: (1 - self.k) * s.source.m,
            )
            for index in range(1, self.n_stages + 1)
        ]
        return CompositeSystem(
            self.label,
            elements=[StageElement('fine'), *stages, StageElement('coarse')],
            interactions=interactions,
        )
