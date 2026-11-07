"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentResult, alignment
from ._autocatalytic import AutocatalyticSetAnalysis
from ._automata import AutomatonResult, automaton
from ._base import BaseAnalysis, bind_analysis
from ._cell_complex import CellComplexAnalysis
from ._clusters import ClusterAnalysis
from ._coarse_grain import CoarseGrainAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityResult, complexity
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationResult, correlation
from ._defects import DefectAnalysis
from ._discovery import DiscoveryAnalysis
from ._displacement import DisplacementAnalysis
from ._distribution import DistributionResult, distribution
from ._domains import DomainsResult, domains
from ._elementary_modes import ElementaryModesResult, elementary_modes
from ._embedding import EmbeddingResult, embedding
from ._emergence import EmergenceResult, emergence
from ._entropy import EntropyResult, entropy
from ._ergodicity import ErgodicityResult, ergodicity
from ._events import EventsResult, events
from ._fit import FitLawResult, fit_law
from ._information import InformationResult, information
from ._invariants import InvariantsResult, invariants
from ._irreversibility import IrreversibilityResult, irreversibility
from ._kramers_moyal import KramersMoyalResult, kramers_moyal
from ._large_deviations import LargeDeviationsResult, large_deviations
from ._linearization import LinearizationResult, linearization
from ._local_information import LocalInformationResult, local_information
from ._loops import LoopsResult, loops
from ._lyapunov import LyapunovResult, lyapunov
from ._markov import MarkovResult, markov
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifsResult, motifs
from ._neighbors import NeighborsResult, neighbors
from ._networks import NetworkMeasuresResult, network_measures
from ._operator import OperatorMeasuresResult, operator_measures
from ._partial_information import PartialInformationResult, partial_information
from ._passage import PassageResult, passage
from ._recurrence import RecurrenceResult, recurrence
from ._renormalization import RenormalizationResult, renormalization
from ._reservoirs import ReservoirAnalysis
from ._rolling import RollingResult, rolling
from ._scaling import ScalingAnalysis
from ._sections import SectionResult, section
from ._shape import ShapeAnalysis
from ._signatures import SignaturesResult, signatures
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumResult, spectrum
from ._stability import StabilityResult, stability
from ._surrogates import SurrogateTestResult, surrogate_test
from ._symbolic import SymbolicResult, symbolic

__all__: list[str] = [
    'AlignmentResult',
    'AutocatalyticSetAnalysis',
    'AutomatonResult',
    'BaseAnalysis',
    'CellComplexAnalysis',
    'ClusterAnalysis',
    'CoarseGrainAnalysis',
    'CollapseAnalysis',
    'ComplexityResult',
    'CorrelationResult',
    'DefectAnalysis',
    'DiscoveryAnalysis',
    'DisplacementAnalysis',
    'DistributionResult',
    'DomainsResult',
    'ElementaryModesResult',
    'EmbeddingResult',
    'EmergenceResult',
    'EntropyResult',
    'EpsilonMachineAnalysis',
    'ErgodicityResult',
    'EventsResult',
    'FitLawResult',
    'InformationResult',
    'InvariantsResult',
    'IrreversibilityResult',
    'KramersMoyalResult',
    'LargeDeviationsResult',
    'LinearizationResult',
    'LocalInformationResult',
    'LoopsResult',
    'LyapunovResult',
    'MarkovResult',
    'ModeDecompositionAnalysis',
    'MotifsResult',
    'NeighborsResult',
    'NetworkMeasuresResult',
    'OperatorMeasuresResult',
    'PartialInformationResult',
    'PassageResult',
    'RecurrenceResult',
    'RenormalizationResult',
    'ReservoirAnalysis',
    'RollingResult',
    'ScalingAnalysis',
    'SectionResult',
    'ShapeAnalysis',
    'SignaturesResult',
    'SparseIdentificationAnalysis',
    'SpectrumResult',
    'StabilityResult',
    'SurrogateTestResult',
    'SymbolicResult',
    'alignment',
    'automaton',
    'bind_analysis',
    'complexity',
    'correlation',
    'distribution',
    'domains',
    'elementary_modes',
    'embedding',
    'emergence',
    'entropy',
    'ergodicity',
    'events',
    'fit_law',
    'information',
    'invariants',
    'irreversibility',
    'kramers_moyal',
    'large_deviations',
    'linearization',
    'local_information',
    'loops',
    'lyapunov',
    'markov',
    'motifs',
    'neighbors',
    'network_measures',
    'operator_measures',
    'partial_information',
    'passage',
    'recurrence',
    'renormalization',
    'rolling',
    'section',
    'signatures',
    'spectrum',
    'stability',
    'surrogate_test',
    'symbolic',
]
