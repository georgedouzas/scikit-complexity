"""Neural networks: Hopfield's memory, its diluted version, reservoir computers, and Elman's recurrent network."""

from ._diluted import DilutedNeuronElement, diluted_hopfield
from ._echo_state import ReservoirUnitElement, echo_state_network
from ._elman import (
    HiddenUnitElement,
    InputCouplingInteraction,
    InputUnitElement,
    OutputUnitElement,
    ReadoutCouplingInteraction,
    RecurrenceCouplingInteraction,
    elman_network,
)
from ._hopfield import HopfieldUpdateInteraction, NeuronElement, hopfield
from ._next_generation import LorenzFlowElement, next_generation_reservoir
from ._perceptron import BinaryWeightElement, perceptron_satisfiability

__all__: list[str] = [
    'BinaryWeightElement',
    'DilutedNeuronElement',
    'HiddenUnitElement',
    'HopfieldUpdateInteraction',
    'InputCouplingInteraction',
    'InputUnitElement',
    'LorenzFlowElement',
    'NeuronElement',
    'OutputUnitElement',
    'ReadoutCouplingInteraction',
    'RecurrenceCouplingInteraction',
    'ReservoirUnitElement',
    'diluted_hopfield',
    'echo_state_network',
    'elman_network',
    'hopfield',
    'next_generation_reservoir',
    'perceptron_satisfiability',
]
