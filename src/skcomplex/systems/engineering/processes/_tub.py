"""Tub of water, a stock filled by a faucet that opens after five minutes and emptied by a drain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.components.laws.utils import step, where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

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
