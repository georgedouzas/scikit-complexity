"""Neural networks: Hopfield memories, reservoir computers, and Elman's network.

The Hopfield memories include its diluted version, and Elman's network is recurrent.
"""

from ._diluted import DilutedHopfieldSystem, DilutedNeuronElement
from ._echo_state import EchoStateNetworkSystem, ReservoirUnitElement
from ._elman import (
    ElmanNetworkSystem,
    HiddenUnitElement,
    InputCouplingInteraction,
    InputUnitElement,
    OutputUnitElement,
    ReadoutCouplingInteraction,
    RecurrenceCouplingInteraction,
)
from ._hopfield import HopfieldSystem, HopfieldUpdateInteraction, NeuronElement
from ._next_generation import LorenzFlowElement, NextGenerationReservoirSystem
from ._perceptron import (
    BinaryWeightElement,
    CardinalitySpec,
    ClauseSpec,
    PerceptronSatisfiabilitySystem,
    SatisfiabilityInteraction,
)

__all__: list[str] = [
    'BinaryWeightElement',
    'CardinalitySpec',
    'ClauseSpec',
    'DilutedHopfieldSystem',
    'DilutedNeuronElement',
    'EchoStateNetworkSystem',
    'ElmanNetworkSystem',
    'HiddenUnitElement',
    'HopfieldSystem',
    'HopfieldUpdateInteraction',
    'InputCouplingInteraction',
    'InputUnitElement',
    'LorenzFlowElement',
    'NeuronElement',
    'NextGenerationReservoirSystem',
    'OutputUnitElement',
    'PerceptronSatisfiabilitySystem',
    'ReadoutCouplingInteraction',
    'RecurrenceCouplingInteraction',
    'ReservoirUnitElement',
    'SatisfiabilityInteraction',
]
