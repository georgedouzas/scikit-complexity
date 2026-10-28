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
from ._base import BaseAnalysis
from ._collapse import CollapseAnalysis
from ._complexity import ComplexityEntropy, EnsembleEntropy, complexity_entropy, ensemble_entropy
from ._computational_mechanics import EpsilonMachineAnalysis
from ._correlation import CorrelationAnalysis
from ._criticality import (
    AvalancheLaws,
    Avalanches,
    CracklingNoise,
    Sweep,
    avalanches,
    branching_ratio,
    crackling_noise,
    fit_avalanches,
    fit_order_exponent,
    order_parameter_sweep,
    replica_overlap,
)
from ._displacement import DisplacementAnalysis
from ._domains import DomainAnalysis
from ._early_warning import EarlyWarningAnalysis
from ._embedding import EmbeddingAnalysis
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import EventAnalysis
from ._extremes import (
    ExtremeValueFit,
    GoodnessOfFit,
    StableFit,
    compare_with_law,
    fit_extreme_values,
    fit_stable,
    record_counts,
)
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
from ._percolation import Percolation, cluster_sizes, percolation, spanning_probability
from ._quantum import QuantumAnalysis
from ._recurrence import RecurrenceAnalysis
from ._reservoirs import ReservoirAnalysis
from ._scaling import ScalingAnalysis, binder_crossings, binder_cumulant
from ._sections import SectionAnalysis
from ._signatures import Signature, signature_analysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spatial import Clusters, label_clusters
from ._spectrum import SpectrumAnalysis
from ._stability import StabilityAnalysis
from ._statistics import PowerLawFit, ccdf, fit_power_law, gini, hill, histogram
from ._surrogates import SurrogateTest, surrogate_test
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'ANALYSES',
    'AnalysisEntry',
    'Applicable',
    'Automaton',
    'AutomatonAnalysis',
    'AvalancheLaws',
    'Avalanches',
    'BaseAnalysis',
    'Clusters',
    'CollapseAnalysis',
    'ComplexityEntropy',
    'CorrelationAnalysis',
    'CracklingNoise',
    'DataKind',
    'DisplacementAnalysis',
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
    'ExtremeValueFit',
    'FluctuationAnalysis',
    'GoodnessOfFit',
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
    'Percolation',
    'PowerLawFit',
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
    'StableFit',
    'Subshift',
    'SurrogateTest',
    'Sweep',
    'SymbolicAnalysis',
    'analyses_for',
    'analysis',
    'as_graph',
    'avalanches',
    'binder_crossings',
    'binder_cumulant',
    'branching_ratio',
    'ccdf',
    'cluster_sizes',
    'coarse_grain',
    'compare_with_law',
    'complexity_entropy',
    'crackling_noise',
    'data_kinds',
    'ensemble_entropy',
    'ensemble_of',
    'fields_of',
    'fit_avalanches',
    'fit_extreme_values',
    'fit_order_exponent',
    'fit_power_law',
    'fit_stable',
    'gini',
    'hill',
    'histogram',
    'label_clusters',
    'mittag_leffler',
    'networks_of',
    'order_parameter_sweep',
    'path_of',
    'percolation',
    'record_counts',
    'replica_overlap',
    'series_of',
    'signature_analysis',
    'snapshots_of',
    'spanning_probability',
    'surrogate_test',
]
