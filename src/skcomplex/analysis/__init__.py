"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentAnalysis
from ._automata import AutomatonAnalysis
from ._base import BaseAnalysis
from ._clusters import ClusterAnalysis
from ._coarse_grain import CoarseGrainAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityAnalysis
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationAnalysis
from ._defects import DefectAnalysis
from ._discovery import DiscoveryAnalysis
from ._displacement import DisplacementAnalysis
from ._distribution import DistributionAnalysis
from ._domains import DomainAnalysis
from ._embedding import EmbeddingAnalysis
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import EventAnalysis
from ._fit import FitAnalysis
from ._information import InformationAnalysis
from ._invariants import InvariantAnalysis
from ._irreversibility import IrreversibilityAnalysis
from ._kramers_moyal import KramersMoyalAnalysis
from ._large_deviations import LargeDeviationAnalysis
from ._linearization import LinearizationAnalysis
from ._local_information import LocalInformationAnalysis
from ._loops import LoopAnalysis
from ._lyapunov import LyapunovAnalysis
from ._markov import MarkovAnalysis
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifAnalysis
from ._neighbors import NeighborAnalysis
from ._networks import NetworkAnalysis
from ._operator import OperatorAnalysis
from ._partial_information import PartialInformationAnalysis
from ._passage import PassageAnalysis
from ._recurrence import RecurrenceAnalysis
from ._reservoirs import ReservoirAnalysis
from ._rolling import RollingAnalysis
from ._scaling import ScalingAnalysis
from ._sections import SectionAnalysis
from ._signatures import SignatureAnalysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumAnalysis
from ._stability import StabilityAnalysis
from ._surrogates import SurrogateTestAnalysis
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'AlignmentAnalysis',
    'AutomatonAnalysis',
    'BaseAnalysis',
    'ClusterAnalysis',
    'CoarseGrainAnalysis',
    'CollapseAnalysis',
    'ComplexityAnalysis',
    'CorrelationAnalysis',
    'DefectAnalysis',
    'DiscoveryAnalysis',
    'DisplacementAnalysis',
    'DistributionAnalysis',
    'DomainAnalysis',
    'EmbeddingAnalysis',
    'EmergenceAnalysis',
    'EntropyAnalysis',
    'EpsilonMachineAnalysis',
    'ErgodicityAnalysis',
    'EventAnalysis',
    'FitAnalysis',
    'InformationAnalysis',
    'InvariantAnalysis',
    'IrreversibilityAnalysis',
    'KramersMoyalAnalysis',
    'LargeDeviationAnalysis',
    'LinearizationAnalysis',
    'LocalInformationAnalysis',
    'LoopAnalysis',
    'LyapunovAnalysis',
    'MarkovAnalysis',
    'ModeDecompositionAnalysis',
    'MotifAnalysis',
    'NeighborAnalysis',
    'NetworkAnalysis',
    'OperatorAnalysis',
    'PartialInformationAnalysis',
    'PassageAnalysis',
    'RecurrenceAnalysis',
    'ReservoirAnalysis',
    'RollingAnalysis',
    'ScalingAnalysis',
    'SectionAnalysis',
    'SignatureAnalysis',
    'SparseIdentificationAnalysis',
    'SpectrumAnalysis',
    'StabilityAnalysis',
    'SurrogateTestAnalysis',
    'SymbolicAnalysis',
]
