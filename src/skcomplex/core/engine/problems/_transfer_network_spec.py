"""Define the transfer network spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ._loop_term_spec import LoopTermSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class _CompiledDelay:
    """Delay or smoothing of a compiled system.

    It holds its call in the laws and its output, read from hidden states.
    """

    call: Expression
    output: Expression
    references: Mapping[Expression, ReferenceSpec]


@dataclass(frozen=True)
class _CompiledFlow:
    """Transfer of a compiled system between two holders."""

    path: str
    source: str | None
    sink: str | None
    source_label: str
    sink_label: str
    quantity: str | None


@dataclass(frozen=True)
class _Recorded:
    """Expression a trajectory records at every frame.

    It holds the references of its symbols.
    """

    path: str
    expression: Expression
    references: Mapping[Expression, ReferenceSpec]


@dataclass(frozen=True)
class TransferNetworkSpec(BaseSpec):
    """A spec of what flows between the stocks of a system and what runs record.

    A system of stocks and flows, such as water that drains from one tank into another, keeps the quantities every flow
    moves, the flows themselves, the values a trajectory records, and the terms of its feedback graph. Analyses read it
    to check that a quantity is conserved and to find feedback loops. It holds the flows of water between tanks, money
    between accounts, and people between the compartments of an epidemic.

    Read more in the user guide, under Analyse.

    Args:
        quantities:
            The names of the states that hold each quantity.

        flows:
            The flows.

        recorded:
            The flows and the running values of objectives that trajectories record.

        boundary:
            The hidden state that accumulates the net inflow from the environment of each quantity with boundary flows.

        terms:
            The terms of the feedback graph.

        discrete:
            Whether flow amounts are per step rather than rates.

        delays:
            The delays and smoothings of the laws, in the order the compiler replaced them by hidden states.

    See Also:
        LoopTermSpec: One term of the feedback graph.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> network = LogisticGrowthSystem().prepare().simulate(until=0).problem.transfer_network
        >>> network.discrete, network.terms[0].path
        (False, 'population.x')
    """

    quantities: Mapping[str, tuple[str, ...]]
    flows: tuple[_CompiledFlow, ...] = ()
    recorded: tuple[_Recorded, ...] = ()
    boundary: Mapping[str, str] = field(default_factory=dict)
    terms: tuple[LoopTermSpec, ...] = ()
    discrete: bool = False
    delays: tuple[_CompiledDelay, ...] = ()
