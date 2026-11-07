"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._alignment import AlignmentAnalysis
from ._autocatalytic import AutocatalyticSetAnalysis
from ._automata import AutomatonResult, automaton
from ._base import BaseAnalysis, bind_analysis
from ._cell_complex import CellComplexAnalysis
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
from ._elementary_modes import ElementaryModesResult, elementary_modes
from ._embedding import EmbeddingAnalysis
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import EventAnalysis
from ._fit import FitAnalysis
from ._information import InformationAnalysis
from ._invariants import InvariantsResult, invariants
from ._irreversibility import IrreversibilityAnalysis
from ._kramers_moyal import KramersMoyalAnalysis
from ._large_deviations import LargeDeviationAnalysis
from ._linearization import LinearizationResult, linearization
from ._local_information import LocalInformationAnalysis
from ._loops import LoopsResult, loops
from ._lyapunov import LyapunovResult, lyapunov
from ._markov import MarkovAnalysis
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifAnalysis
from ._neighbors import NeighborAnalysis
from ._networks import NetworkAnalysis
from ._operator import OperatorMeasuresResult, operator_measures
from ._partial_information import PartialInformationAnalysis
from ._passage import PassageAnalysis
from ._recurrence import RecurrenceAnalysis
from ._renormalization import RenormalizationAnalysis
from ._reservoirs import ReservoirAnalysis
from ._rolling import RollingResult, rolling
from ._scaling import ScalingAnalysis
from ._sections import SectionResult, section
from ._shape import ShapeAnalysis
from ._signatures import SignatureAnalysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumAnalysis
from ._stability import StabilityResult, stability
from ._surrogates import SurrogateTestAnalysis
from ._symbolic import SymbolicResult, symbolic

__all__: list[str] = [
    'AlignmentAnalysis',
    'AutocatalyticSetAnalysis',
    'AutomatonResult',
    'BaseAnalysis',
    'CellComplexAnalysis',
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
    'ElementaryModesResult',
    'EmbeddingAnalysis',
    'EmergenceAnalysis',
    'EntropyAnalysis',
    'EpsilonMachineAnalysis',
    'ErgodicityAnalysis',
    'EventAnalysis',
    'FitAnalysis',
    'InformationAnalysis',
    'InvariantsResult',
    'IrreversibilityAnalysis',
    'KramersMoyalAnalysis',
    'LargeDeviationAnalysis',
    'LinearizationResult',
    'LocalInformationAnalysis',
    'LoopsResult',
    'LyapunovResult',
    'MarkovAnalysis',
    'ModeDecompositionAnalysis',
    'MotifAnalysis',
    'NeighborAnalysis',
    'NetworkAnalysis',
    'OperatorMeasuresResult',
    'PartialInformationAnalysis',
    'PassageAnalysis',
    'RecurrenceAnalysis',
    'RenormalizationAnalysis',
    'ReservoirAnalysis',
    'RollingResult',
    'ScalingAnalysis',
    'SectionResult',
    'ShapeAnalysis',
    'SignatureAnalysis',
    'SparseIdentificationAnalysis',
    'SpectrumAnalysis',
    'StabilityResult',
    'SurrogateTestAnalysis',
    'SymbolicResult',
    'automaton',
    'bind_analysis',
    'elementary_modes',
    'invariants',
    'linearization',
    'loops',
    'lyapunov',
    'operator_measures',
    'rolling',
    'section',
    'stability',
    'symbolic',
]
