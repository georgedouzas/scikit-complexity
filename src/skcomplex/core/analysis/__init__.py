"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentResult, alignment
from ._autocatalytic import AutocatalyticSetsResult, autocatalytic_sets
from ._automata import AutomatonResult, automaton
from ._base import BaseAnalysis, bind_analysis
from ._cell_complex import CellComplexResult, cell_complex
from ._clusters import ClustersResult, clusters
from ._coarse_grain import CoarseGrainResult, coarse_grain
from ._collapse import DataCollapseResult, data_collapse
from ._complexity import ComplexityResult, complexity
from ._computational_mechanics import EpsilonMachineResult, epsilon_machine
from ._correlation import CorrelationResult, correlation
from ._defects import DefectsResult, defects
from ._discovery import DiscoveryAnalysis
from ._displacement import DisplacementResult, displacement
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
from ._mode_decomposition import ModeDecompositionResult, mode_decomposition
from ._motifs import MotifsResult, motifs
from ._neighbors import NeighborsResult, neighbors
from ._networks import NetworkMeasuresResult, network_measures
from ._operator import OperatorMeasuresResult, operator_measures
from ._partial_information import PartialInformationResult, partial_information
from ._passage import PassageResult, passage
from ._recurrence import RecurrenceResult, recurrence
from ._renormalization import RenormalizationResult, renormalization
from ._reservoirs import ReservoirResult, reservoir
from ._rolling import RollingResult, rolling
from ._scaling import ScalingResult, scaling
from ._sections import SectionResult, section
from ._shape import ChainShapeResult, chain_shape
from ._signatures import SignaturesResult, signatures
from ._sparse_identification import SparseIdentificationResult, sparse_identification
from ._spectrum import SpectrumResult, spectrum
from ._stability import StabilityResult, stability
from ._surrogates import SurrogateTestResult, surrogate_test
from ._symbolic import SymbolicResult, symbolic

__all__: list[str] = [
    'AlignmentResult',
    'AutocatalyticSetsResult',
    'AutomatonResult',
    'BaseAnalysis',
    'CellComplexResult',
    'ChainShapeResult',
    'ClustersResult',
    'CoarseGrainResult',
    'ComplexityResult',
    'CorrelationResult',
    'DataCollapseResult',
    'DefectsResult',
    'DiscoveryAnalysis',
    'DisplacementResult',
    'DistributionResult',
    'DomainsResult',
    'ElementaryModesResult',
    'EmbeddingResult',
    'EmergenceResult',
    'EntropyResult',
    'EpsilonMachineResult',
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
    'ModeDecompositionResult',
    'MotifsResult',
    'NeighborsResult',
    'NetworkMeasuresResult',
    'OperatorMeasuresResult',
    'PartialInformationResult',
    'PassageResult',
    'RecurrenceResult',
    'RenormalizationResult',
    'ReservoirResult',
    'RollingResult',
    'ScalingResult',
    'SectionResult',
    'SignaturesResult',
    'SparseIdentificationResult',
    'SpectrumResult',
    'StabilityResult',
    'SurrogateTestResult',
    'SymbolicResult',
    'alignment',
    'autocatalytic_sets',
    'automaton',
    'bind_analysis',
    'cell_complex',
    'chain_shape',
    'clusters',
    'coarse_grain',
    'complexity',
    'correlation',
    'data_collapse',
    'defects',
    'displacement',
    'distribution',
    'domains',
    'elementary_modes',
    'embedding',
    'emergence',
    'entropy',
    'epsilon_machine',
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
    'mode_decomposition',
    'motifs',
    'neighbors',
    'network_measures',
    'operator_measures',
    'partial_information',
    'passage',
    'recurrence',
    'renormalization',
    'reservoir',
    'rolling',
    'scaling',
    'section',
    'signatures',
    'sparse_identification',
    'spectrum',
    'stability',
    'surrogate_test',
    'symbolic',
]
