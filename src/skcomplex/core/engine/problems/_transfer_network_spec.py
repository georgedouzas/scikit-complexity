"""Define the transfer network spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec
from ...specs._compiled_flow_spec import _CompiledFlowSpec
from ._compiled_delay_spec import _CompiledDelaySpec
from ._loop_term_spec import LoopTermSpec
from ._recorded_spec import _RecordedSpec


@dataclass(frozen=True)
class TransferNetworkSpec(BaseSpec):
    """A spec of the quantities, flows, and recorded values of a compiled system.

    Args:
        quantities:
            The names of the states that hold each quantity.

        flows:
            The flows.

        recorded:
            The flows and the running values of objectives that trajectories record.

        boundary:
            The hidden state that accumulates the net inflow from the environment of each quantity with boundary
            flows.

        terms:
            The terms of the feedback graph.

        discrete:
            Whether flow amounts are per step rather than rates.

        delays:
            The delays and smoothings of the laws, in the order the compiler replaced them by hidden states.
    """

    quantities: Mapping[str, tuple[str, ...]]
    flows: tuple[_CompiledFlowSpec, ...] = ()
    recorded: tuple[_RecordedSpec, ...] = ()
    boundary: Mapping[str, str] = field(default_factory=dict)
    terms: tuple[LoopTermSpec, ...] = ()
    discrete: bool = False
    delays: tuple[_CompiledDelaySpec, ...] = ()
