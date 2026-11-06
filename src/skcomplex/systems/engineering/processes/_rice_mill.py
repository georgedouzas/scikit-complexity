"""Rice mill that buys seasonal paddy, mills it into rice, husk, and bran, and adjusts its capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import delay3, minimum, smooth
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

MONTHS_PER_YEAR = 12.0


HUSK_SHARE = 0.2


BRAN_SHARE = 0.08


class RiceMillElement(Element):
    """Rice mill that buys seasonal paddy, mills it into rice, husk, and bran, and adjusts its capacity."""

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
        """Return a seasonal value that oscillates about its mean with a relative amplitude and a period."""
        return mean * (1 + amplitude * sin(2 * math.pi * t / period))

    @staticmethod
    def _procurement(s: SimpleNamespace) -> Expression:
        """Return the procurement of paddy, the seasonal harvest after the procurement delay."""
        return delay3(RiceMillElement._season(s.harvest, s.seasonality, s.t, MONTHS_PER_YEAR), s.procurement_delay)

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
        """Return the procurement, milling, and sales of a rice mill, and the adjustment of its milling capacity."""
        milling = RiceMillElement._milling(s)
        losses = 1 - s.outturn - s.husk_share - s.bran_share
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
                (smooth(RiceMillElement._indicated_milling(s), s.perception_time) - s.capacity) / s.expansion_time,
            ),
        ]
