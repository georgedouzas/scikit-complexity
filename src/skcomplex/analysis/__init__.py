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
from ._automata import (
    Automaton,
    LinearAutomaton,
    Subshift,
    compose,
    configuration_entropy,
    difference_pattern,
    garden_of_eden,
    inverse,
    is_injective,
    is_pre_injective,
    is_surjective,
    spatial_entropy_rates,
)
from ._base import BaseAnalysis
from ._bifurcation import OrbitDiagram, orbit_diagram
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
from ._invariants import (
    CorrelationDimension,
    correlation_dimension,
    kaplan_yorke_dimension,
    kolmogorov_sinai_entropy,
    largest_lyapunov_exponent,
)
from ._irreversibility import IrreversibilityAnalysis
from ._kinds import ANALYSES, AnalysisEntry, DataKind, analysis
from ._kramers_moyal import KramersMoyalAnalysis
from ._large_deviations import LargeDeviationAnalysis
from ._linearization import StateSpace, controllability, linearize, lqr, observability
from ._local_information import LocalInformationAnalysis
from ._loops import FlowMatrix, Loop, LoopDominance, Loops, conservation, flow_matrix, loop_dominance, loops
from ._lyapunov import lyapunov_spectrum
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
from ._quantum import Spectrum, entanglement_entropy, ground_state, hamiltonian_of, purity, spectrum
from ._recurrence import Recurrence, recurrence_matrix, recurrence_quantification
from ._reservoirs import ReservoirAnalysis
from ._scaling import Collapse, binder_crossings, binder_cumulant, scaling_collapse
from ._sections import Section, poincare_section, stroboscopic
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
from ._stability import Dynamics, Stability, divergence, dynamics_of, fixed_points, jacobian, nullclines, stability
from ._statistics import PowerLawFit, ccdf, fit_power_law, gini, hill, histogram
from ._surrogates import SurrogateTest, surrogate_test
from ._symbolic import equations_of, solve_symbolically

__all__: list[str] = [
    'ANALYSES',
    'AgingCollapse',
    'AnalysisEntry',
    'Applicable',
    'Automaton',
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
    'Dynamics',
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
    'FlowMatrix',
    'Fluctuation',
    'FluctuationAnalysis',
    'GoodnessOfFit',
    'InformationAnalysis',
    'IrreversibilityAnalysis',
    'JumpPath',
    'KramersMoyalAnalysis',
    'LargeDeviationAnalysis',
    'LinearAutomaton',
    'LocalInformationAnalysis',
    'Loop',
    'LoopDominance',
    'Loops',
    'MarkovAnalysis',
    'MixingAnalysis',
    'ModeDecompositionAnalysis',
    'MotifAnalysis',
    'NeighborAnalysis',
    'NetworkAnalysis',
    'OrbitDiagram',
    'OrderAnalysis',
    'PartialInformationAnalysis',
    'PassageAnalysis',
    'Percolation',
    'PowerLawFit',
    'Recurrence',
    'ReservoirAnalysis',
    'Section',
    'Series',
    'Signature',
    'Snapshots',
    'SparseIdentificationAnalysis',
    'Spectrum',
    'Stability',
    'StableFit',
    'StateSpace',
    'StructureFactor',
    'Subshift',
    'SurrogateTest',
    'Sweep',
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
    'compose',
    'configuration_entropy',
    'conservation',
    'controllability',
    'correlation_dimension',
    'correlation_function',
    'correlation_length',
    'crackling_noise',
    'data_kinds',
    'delay_embedding',
    'detect_events',
    'detrended_fluctuation',
    'difference_pattern',
    'dispersion_relation',
    'divergence',
    'dynamics_of',
    'early_warning',
    'embed',
    'ensemble_entropy',
    'ensemble_of',
    'entanglement_entropy',
    'equations_of',
    'event_counts',
    'false_nearest_neighbors',
    'fields_of',
    'fit_avalanches',
    'fit_extreme_values',
    'fit_order_exponent',
    'fit_power_law',
    'fit_stable',
    'fixed_points',
    'flow_matrix',
    'garden_of_eden',
    'gini',
    'ground_state',
    'hamiltonian_of',
    'hill',
    'histogram',
    'hurst_exponent',
    'inverse',
    'is_injective',
    'is_pre_injective',
    'is_surjective',
    'jacobian',
    'kaplan_yorke_dimension',
    'kolmogorov_sinai_entropy',
    'label_clusters',
    'laminar_durations',
    'largest_lyapunov_exponent',
    'linearize',
    'loop_dominance',
    'loops',
    'lqr',
    'lyapunov_spectrum',
    'mittag_leffler',
    'networks_of',
    'nullclines',
    'observability',
    'orbit_diagram',
    'order_parameter_sweep',
    'path_of',
    'percolation',
    'poincare_section',
    'power_spectrum',
    'purity',
    'record_counts',
    'recurrence_matrix',
    'recurrence_quantification',
    'replica_overlap',
    'scaling_collapse',
    'series_of',
    'signature_analysis',
    'snapshots_of',
    'solve_symbolically',
    'spanning_probability',
    'spatial_entropy_rates',
    'spectrum',
    'stability',
    'stroboscopic',
    'structure_factor',
    'surrogate_test',
    'two_time_correlation',
]
