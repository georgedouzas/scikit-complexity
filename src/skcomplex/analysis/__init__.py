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
from ._aging import AgingCollapse, DistributionCollapse, aging_collapse, collapse_distributions, two_time_correlation
from ._automata import Automaton, AutomatonAnalysis, LinearAutomaton, Subshift
from ._base import BaseAnalysis
from ._complexity import ComplexityEntropy, EnsembleEntropy, complexity_entropy, ensemble_entropy
from ._computational_mechanics import EpsilonMachineAnalysis
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
from ._embedding import Embedding, delay_embedding, embed, false_nearest_neighbors
from ._emergence import EmergenceAnalysis
from ._entropy import EntropyAnalysis
from ._ergodicity import ErgodicityAnalysis
from ._events import Events, detect_events
from ._extremes import (
    ExtremeValueFit,
    GoodnessOfFit,
    StableFit,
    compare_with_law,
    fit_extreme_values,
    fit_stable,
    record_counts,
)
from ._fields import DispersionRelation, StructureFactor, correlation_length, dispersion_relation, structure_factor
from ._fluctuations import FluctuationAnalysis, event_counts
from ._fractional import mittag_leffler
from ._information import InformationAnalysis
from ._invariants import CorrelationDimension, correlation_dimension
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
from ._passage import PassageAnalysis, laminar_durations
from ._percolation import Percolation, cluster_sizes, percolation, spanning_probability
from ._quantum import QuantumAnalysis
from ._recurrence import Recurrence, recurrence_matrix, recurrence_quantification
from ._reservoirs import ReservoirAnalysis
from ._scaling import Collapse, binder_crossings, binder_cumulant, scaling_collapse
from ._sections import SectionAnalysis
from ._series import (
    EarlyWarning,
    Fluctuation,
    autocorrelation,
    autocorrelation_time,
    detrended_fluctuation,
    early_warning,
    hurst_exponent,
    power_spectrum,
)
from ._signatures import Signature, signature_analysis
from ._sparse_identification import SparseIdentificationAnalysis
from ._spatial import Clusters, box_counting_dimension, correlation_function, label_clusters
from ._stability import StabilityAnalysis
from ._statistics import PowerLawFit, ccdf, fit_power_law, gini, hill, histogram
from ._surrogates import SurrogateTestAnalysis
from ._symbolic import SymbolicAnalysis

__all__: list[str] = [
    'ANALYSES',
    'AgingCollapse',
    'AnalysisEntry',
    'Applicable',
    'Automaton',
    'AutomatonAnalysis',
    'AvalancheLaws',
    'Avalanches',
    'BaseAnalysis',
    'Clusters',
    'Collapse',
    'ComplexityEntropy',
    'CorrelationDimension',
    'CracklingNoise',
    'DataKind',
    'DispersionRelation',
    'DisplacementAnalysis',
    'DistributionCollapse',
    'DomainAnalysis',
    'EarlyWarning',
    'Embedding',
    'EmergenceAnalysis',
    'Ensemble',
    'EnsembleEntropy',
    'EntropyAnalysis',
    'EpsilonMachineAnalysis',
    'ErgodicityAnalysis',
    'Events',
    'ExtremeValueFit',
    'Fluctuation',
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
    'Recurrence',
    'ReservoirAnalysis',
    'SectionAnalysis',
    'Series',
    'Signature',
    'Snapshots',
    'SparseIdentificationAnalysis',
    'StabilityAnalysis',
    'StableFit',
    'StructureFactor',
    'Subshift',
    'SurrogateTestAnalysis',
    'Sweep',
    'SymbolicAnalysis',
    'aging_collapse',
    'analyses_for',
    'analysis',
    'as_graph',
    'autocorrelation',
    'autocorrelation_time',
    'avalanches',
    'binder_crossings',
    'binder_cumulant',
    'box_counting_dimension',
    'branching_ratio',
    'ccdf',
    'cluster_sizes',
    'coarse_grain',
    'collapse_distributions',
    'compare_with_law',
    'complexity_entropy',
    'correlation_dimension',
    'correlation_function',
    'correlation_length',
    'crackling_noise',
    'data_kinds',
    'delay_embedding',
    'detect_events',
    'detrended_fluctuation',
    'dispersion_relation',
    'early_warning',
    'embed',
    'ensemble_entropy',
    'ensemble_of',
    'event_counts',
    'false_nearest_neighbors',
    'fields_of',
    'fit_avalanches',
    'fit_extreme_values',
    'fit_order_exponent',
    'fit_power_law',
    'fit_stable',
    'gini',
    'hill',
    'histogram',
    'hurst_exponent',
    'label_clusters',
    'laminar_durations',
    'mittag_leffler',
    'networks_of',
    'order_parameter_sweep',
    'path_of',
    'percolation',
    'power_spectrum',
    'record_counts',
    'recurrence_matrix',
    'recurrence_quantification',
    'replica_overlap',
    'scaling_collapse',
    'series_of',
    'signature_analysis',
    'snapshots_of',
    'spanning_probability',
    'structure_factor',
    'two_time_correlation',
]
