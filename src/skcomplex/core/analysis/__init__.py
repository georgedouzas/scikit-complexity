"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentResult, alignment
from ._autocatalytic import AutocatalyticSetAnalysis
from ._automata import AutomatonAnalysis
from ._base import BaseAnalysis, bind_analysis
from ._cell_complex import CellComplexAnalysis
from ._clusters import ClusterAnalysis
from ._coarse_grain import CoarseGrainAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityAnalysis
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationResult, correlation
from ._defects import DefectAnalysis
from ._discovery import DiscoveryAnalysis
from ._displacement import DisplacementAnalysis
from ._distribution import DistributionResult, distribution
from ._domains import DomainsResult, domains
from ._elementary_modes import ElementaryModeAnalysis
from ._embedding import EmbeddingAnalysis
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import EventAnalysis
from ._fit import FitLawResult, fit_law
from ._information import InformationAnalysis
from ._invariants import InvariantAnalysis
from ._irreversibility import IrreversibilityAnalysis
from ._kramers_moyal import KramersMoyalAnalysis
from ._large_deviations import LargeDeviationAnalysis
from ._linearization import LinearizationAnalysis
from ._local_information import LocalInformationAnalysis
from ._loops import LoopAnalysis
from ._lyapunov import LyapunovResult, lyapunov
from ._markov import MarkovAnalysis
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifsResult, motifs
from ._neighbors import NeighborsResult, neighbors
from ._networks import NetworkMeasuresResult, network_measures
from ._operator import OperatorAnalysis
from ._partial_information import PartialInformationAnalysis
from ._passage import PassageAnalysis
from ._recurrence import RecurrenceAnalysis
from ._renormalization import RenormalizationResult, renormalization
from ._reservoirs import ReservoirAnalysis
from ._rolling import RollingResult, rolling
from ._scaling import ScalingAnalysis
from ._sections import SectionAnalysis
from ._shape import ShapeAnalysis
from ._signatures import SignaturesResult, signatures
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumResult, spectrum
from ._stability import StabilityAnalysis
from ._surrogates import SurrogateTestAnalysis
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'AlignmentResult',
    'AutocatalyticSetAnalysis',
    'AutomatonAnalysis',
    'BaseAnalysis',
    'CellComplexAnalysis',
    'ClusterAnalysis',
    'CoarseGrainAnalysis',
    'CollapseAnalysis',
    'ComplexityAnalysis',
    'CorrelationResult',
    'DefectAnalysis',
    'DiscoveryAnalysis',
    'DisplacementAnalysis',
    'DistributionResult',
    'DomainsResult',
    'ElementaryModeAnalysis',
    'EmbeddingAnalysis',
    'EmergenceAnalysis',
    'EntropyAnalysis',
    'EpsilonMachineAnalysis',
    'ErgodicityAnalysis',
    'EventAnalysis',
    'FitLawResult',
    'InformationAnalysis',
    'InvariantAnalysis',
    'IrreversibilityAnalysis',
    'KramersMoyalAnalysis',
    'LargeDeviationAnalysis',
    'LinearizationAnalysis',
    'LocalInformationAnalysis',
    'LoopAnalysis',
    'LyapunovResult',
    'MarkovAnalysis',
    'ModeDecompositionAnalysis',
    'MotifsResult',
    'NeighborsResult',
    'NetworkMeasuresResult',
    'OperatorAnalysis',
    'PartialInformationAnalysis',
    'PassageAnalysis',
    'RecurrenceAnalysis',
    'RenormalizationResult',
    'ReservoirAnalysis',
    'RollingResult',
    'ScalingAnalysis',
    'SectionAnalysis',
    'ShapeAnalysis',
    'SignaturesResult',
    'SparseIdentificationAnalysis',
    'SpectrumResult',
    'StabilityAnalysis',
    'SurrogateTestAnalysis',
    'SymbolicAnalysis',
    'alignment',
    'bind_analysis',
    'correlation',
    'distribution',
    'domains',
    'fit_law',
    'lyapunov',
    'motifs',
    'neighbors',
    'network_measures',
    'renormalization',
    'rolling',
    'signatures',
    'spectrum',
]
