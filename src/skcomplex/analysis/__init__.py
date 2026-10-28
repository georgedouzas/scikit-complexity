"""Analyse systems, trajectories, and measured data by kind of data: series, paths, fields, agents, and networks."""

from ._adapters import (
    Applicable,
    Ensemble,
    JumpPath,
    Series,
    Snapshots,
    analyses_for,
    data_kinds,
    ensemble_of,
    fields_of,
    networks_of,
    path_of,
    series_of,
    snapshots_of,
)
from ._agents import coarse_grain
from ._automata import Automaton, AutomatonAnalysis, LinearAutomaton, Subshift
from ._avalanches import AvalancheAnalysis
from ._base import BaseAnalysis
from ._binder import BinderAnalysis
from ._clusters import ClusterAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityEntropy, EnsembleEntropy, complexity_entropy, ensemble_entropy
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationAnalysis
from ._displacement import DisplacementAnalysis
from ._distribution import DistributionAnalysis
from ._domains import DomainAnalysis
from ._early_warning import EarlyWarningAnalysis
from ._embedding import EmbeddingAnalysis
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import EventAnalysis
from ._fit import FitAnalysis
from ._fluctuations import FluctuationAnalysis
from ._fractional import mittag_leffler
from ._information import InformationAnalysis
from ._irreversibility import IrreversibilityAnalysis
from ._kinds import ANALYSES, AnalysisEntry, DataKind, analysis
from ._kramers_moyal import KramersMoyalAnalysis
from ._large_deviations import LargeDeviationAnalysis
from ._linearization import LinearizationAnalysis
from ._local_information import LocalInformationAnalysis
from ._loops import LoopAnalysis
from ._lyapunov import LyapunovAnalysis
from ._markov import MarkovAnalysis
from ._mixing import MixingAnalysis
from ._mode_decomposition import ModeDecompositionAnalysis
from ._motifs import MotifAnalysis
from ._neighbors import NeighborAnalysis
from ._networks import NetworkAnalysis, as_graph
from ._order import OrderAnalysis
from ._partial_information import PartialInformationAnalysis
from ._passage import PassageAnalysis
from ._quantum import QuantumAnalysis
from ._recurrence import RecurrenceAnalysis
from ._reservoirs import ReservoirAnalysis
from ._scaling import ScalingAnalysis
from ._sections import SectionAnalysis
from ._signatures import Signature, signature_analysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spectrum import SpectrumAnalysis
from ._stability import StabilityAnalysis
from ._surrogates import SurrogateTestAnalysis
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'ANALYSES',
    'AnalysisEntry',
    'Applicable',
    'Automaton',
    'AutomatonAnalysis',
    'AvalancheAnalysis',
    'BaseAnalysis',
    'BinderAnalysis',
    'ClusterAnalysis',
    'CollapseAnalysis',
    'ComplexityEntropy',
    'CorrelationAnalysis',
    'DataKind',
    'DisplacementAnalysis',
    'DistributionAnalysis',
    'DomainAnalysis',
    'EarlyWarningAnalysis',
    'EmbeddingAnalysis',
    'EmergenceAnalysis',
    'Ensemble',
    'EnsembleEntropy',
    'EntropyAnalysis',
    'EpsilonMachineAnalysis',
    'ErgodicityAnalysis',
    'EventAnalysis',
    'FitAnalysis',
    'FluctuationAnalysis',
    'InformationAnalysis',
    'IrreversibilityAnalysis',
    'JumpPath',
    'KramersMoyalAnalysis',
    'LargeDeviationAnalysis',
    'LinearAutomaton',
    'LinearizationAnalysis',
    'LocalInformationAnalysis',
    'LoopAnalysis',
    'LyapunovAnalysis',
    'MarkovAnalysis',
    'MixingAnalysis',
    'ModeDecompositionAnalysis',
    'MotifAnalysis',
    'NeighborAnalysis',
    'NetworkAnalysis',
    'OrderAnalysis',
    'PartialInformationAnalysis',
    'PassageAnalysis',
    'QuantumAnalysis',
    'RecurrenceAnalysis',
    'ReservoirAnalysis',
    'ScalingAnalysis',
    'SectionAnalysis',
    'Series',
    'Signature',
    'Snapshots',
    'SparseIdentificationAnalysis',
    'SpectrumAnalysis',
    'StabilityAnalysis',
    'Subshift',
    'SurrogateTestAnalysis',
    'SymbolicAnalysis',
    'analyses_for',
    'analysis',
    'as_graph',
    'coarse_grain',
    'complexity_entropy',
    'data_kinds',
    'ensemble_entropy',
    'ensemble_of',
    'fields_of',
    'mittag_leffler',
    'networks_of',
    'path_of',
    'series_of',
    'signature_analysis',
    'snapshots_of',
]
