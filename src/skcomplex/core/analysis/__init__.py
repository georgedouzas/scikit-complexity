"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentAnalysis
from ._autocatalytic import AutocatalyticSetAnalysis
from ._automata import AutomatonAnalysis
from ._base import BaseAnalysis, bind_analysis
from ._cell_complex import CellComplexAnalysis
from ._clusters import ClusterAnalysis
from ._coarse_grain import CoarseGrainAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityResult, complexity
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationAnalysis
from ._defects import DefectAnalysis
from ._discovery import DiscoveryAnalysis
from ._displacement import DisplacementAnalysis
from ._distribution import DistributionAnalysis
from ._domains import DomainAnalysis
from ._elementary_modes import ElementaryModeAnalysis
from ._embedding import EmbeddingResult, embedding
from ._emergence import EmergenceResult, emergence
from ._entropy import EntropyResult, entropy
from ._ergodicity import ErgodicityResult, ergodicity
from ._events import EventsResult, events
from ._fit import FitAnalysis
from ._information import InformationResult, information
from ._invariants import InvariantAnalysis
from ._irreversibility import IrreversibilityResult, irreversibility
from ._kramers_moyal import KramersMoyalResult, kramers_moyal
from ._large_deviations import LargeDeviationsResult, large_deviations
from ._linearization import LinearizationAnalysis
from ._local_information import LocalInformationResult, local_information
from ._loops import LoopAnalysis
from ._lyapunov import LyapunovResult, lyapunov
from ._markov import MarkovResult, markov
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifAnalysis
from ._neighbors import NeighborAnalysis
from ._networks import NetworkAnalysis
from ._operator import OperatorAnalysis
from ._partial_information import PartialInformationResult, partial_information
from ._passage import PassageResult, passage
from ._recurrence import RecurrenceResult, recurrence
from ._renormalization import RenormalizationAnalysis
from ._reservoirs import ReservoirAnalysis
from ._rolling import RollingResult, rolling
from ._scaling import ScalingAnalysis
from ._sections import SectionAnalysis
from ._shape import ShapeAnalysis
from ._signatures import SignatureAnalysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumAnalysis
from ._stability import StabilityAnalysis
from ._surrogates import SurrogateTestResult, surrogate_test
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'AlignmentAnalysis',
    'AutocatalyticSetAnalysis',
    'AutomatonAnalysis',
    'BaseAnalysis',
    'CellComplexAnalysis',
    'ClusterAnalysis',
    'CoarseGrainAnalysis',
    'CollapseAnalysis',
    'ComplexityResult',
    'CorrelationAnalysis',
    'DefectAnalysis',
    'DiscoveryAnalysis',
    'DisplacementAnalysis',
    'DistributionAnalysis',
    'DomainAnalysis',
    'ElementaryModeAnalysis',
    'EmbeddingResult',
    'EmergenceResult',
    'EntropyResult',
    'EpsilonMachineAnalysis',
    'ErgodicityResult',
    'EventsResult',
    'FitAnalysis',
    'InformationResult',
    'InvariantAnalysis',
    'IrreversibilityResult',
    'KramersMoyalResult',
    'LargeDeviationsResult',
    'LinearizationAnalysis',
    'LocalInformationResult',
    'LoopAnalysis',
    'LyapunovResult',
    'MarkovResult',
    'ModeDecompositionAnalysis',
    'MotifAnalysis',
    'NeighborAnalysis',
    'NetworkAnalysis',
    'OperatorAnalysis',
    'PartialInformationResult',
    'PassageResult',
    'RecurrenceResult',
    'RenormalizationAnalysis',
    'ReservoirAnalysis',
    'RollingResult',
    'ScalingAnalysis',
    'SectionAnalysis',
    'ShapeAnalysis',
    'SignatureAnalysis',
    'SparseIdentificationAnalysis',
    'SpectrumAnalysis',
    'StabilityAnalysis',
    'SurrogateTestResult',
    'SymbolicAnalysis',
    'bind_analysis',
    'complexity',
    'embedding',
    'emergence',
    'entropy',
    'ergodicity',
    'events',
    'information',
    'irreversibility',
    'kramers_moyal',
    'large_deviations',
    'local_information',
    'lyapunov',
    'markov',
    'partial_information',
    'passage',
    'recurrence',
    'rolling',
    'surrogate_test',
]
