"""Process systems: stocks of material that flows in, out, and between stages."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import step, where
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class StageElement(Element):
    """Stage of a process, a stock of material `m` with a constant feed."""

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


INFLOW_START = 5.0


class TubElement(Element):
    """Tub of water, a stock filled by a faucet that opens after five minutes and emptied by a drain."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        water: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='water')] = 50.0,
        inflow: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        outflow: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.water = water
        self.inflow = inflow
        self.outflow = outflow

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the faucet and drain flows of the tub."""
        return [
            TransferLaw(Source(), s.water, s.inflow * step(s.t, INFLOW_START), label='faucet'),
            TransferLaw(s.water, Sink(), where(s.water > 0, s.outflow, 0), label='drain'),
        ]


@register_system(
    field='engineering/processes',
    tags=('stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 10.0, 'dt': 0.5},
)
def bathtub(water: float = 50.0, *, inflow: float = 5.0, outflow: float = 5.0) -> System:
    """Return a bathtub, one stock of water with an inflow from the faucet and an outflow through the drain.

    The drain empties the tub at `outflow` gallons per minute while there is water, and the faucet opens to `inflow`
    gallons per minute after five minutes. With equal flows the level falls for five minutes and then stays constant.

    Args:
        water:
            The initial water in the tub, in gallons.

        inflow:
            The inflow after five minutes, in gallons per minute.

        outflow:
            The outflow, in gallons per minute.

    Returns:
        system:
            The system, with the water `tub.water` in gallons and the flows `tub.faucet` and `tub.drain` in gallons
            per minute, over minutes.

    Examples:
        >>> from skcomplex.systems.engineering.processes import bathtub
        >>> bathtub(water=20.0).elements[0].water
        20.0
    """
    return System(elements=[TubElement('tub', water=water, inflow=inflow, outflow=outflow)])


@register_system(
    field='engineering/processes',
    tags=('separation', 'cascade', 'stocks and flows', 'ode'),
    references=('Barsky 2020',),
    defaults={'until': 50.0, 'dt': 0.5},
)
def cascade_classifier(
    n_stages: int = 10,
    *,
    k: float = 0.6,
    feed_stage: int | None = None,
    feed: float = 1.0,
) -> System:
    """Return a cascade classifier, a column of stages that separates a narrow size class of particles by an air flow.

    Material leaves every stage at unit rate, a fraction `k` upwards and `1 - k` downwards. The top stage releases
    into the fine product and the bottom stage into the coarse product. Material of one size class is fed into one
    stage at a constant rate. At steady state the share of the feed that ends in the fine product is
    `F_f = (1 - ((1 - k) / k)^(z + 1 - i)) / (1 - ((1 - k) / k)^(z + 1))` for `z` stages and feeding at stage `i`
    counted from the top, Barsky's fractional extraction.

    Args:
        n_stages:
            The number of stages `z`.

        k:
            The distribution coefficient, the fraction of material moving up from a stage.

        feed_stage:
            The stage that receives the feed, counted from 1 at the top, or `None` for the middle stage.

        feed:
            The feed rate.

    Returns:
        system:
            The system, with the material `stage_<i>.m` of each stage from the top, and the accumulated products
            `fine.m` and `coarse.m`.

    Examples:
        >>> from skcomplex.systems.engineering.processes import cascade_classifier
        >>> len(cascade_classifier(n_stages=4).interactions)
        8
    """
    entry = (n_stages + 1) // 2 if feed_stage is None else feed_stage
    stages = [StageElement(f'stage_{index}', feed=feed if index == entry else 0.0) for index in range(1, n_stages + 1)]
    labels = ['fine', *(stage.label for stage in stages), 'coarse']
    interactions = [
        TransferInteraction(
            f'up_{index}',
            source=labels[index],
            target=labels[index - 1],
            state='m',
            amount=lambda s: k * s.source.m,
        )
        for index in range(1, n_stages + 1)
    ] + [
        TransferInteraction(
            f'down_{index}',
            source=labels[index],
            target=labels[index + 1],
            state='m',
            amount=lambda s: (1 - k) * s.source.m,
        )
        for index in range(1, n_stages + 1)
    ]
    return System(elements=[StageElement('fine'), *stages, StageElement('coarse')], interactions=interactions)
