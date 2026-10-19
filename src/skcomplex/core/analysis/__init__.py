"""Analyse systems and trajectories: equations, exact solutions, stability, chaos, statistics, and information."""

from ._bifurcation import BasinStability, OrbitDiagram, basin_stability, orbit_diagram
from ._events import Events, detect_events
from ._information import (
    block_entropies,
    block_entropy,
    correlation_information,
    entropy_rate,
    excess_entropy,
    kl_divergence,
    lempel_ziv_complexity,
    mutual_information,
    permutation_entropy,
    shannon_entropy,
    symbolize,
    transfer_entropy,
)
from ._lyapunov import lyapunov_spectrum
from ._networks import NetworkSummary, as_graph, centrality, degrees, summarize_network
from ._sections import Section, poincare_section, stroboscopic
from ._series import (
    Displacement,
    EarlyWarning,
    autocorrelation,
    early_warning,
    hurst_exponent,
    mean_squared_displacement,
    power_spectrum,
)
from ._spatial import Clusters, box_counting_dimension, correlation_function, label_clusters
from ._stability import Dynamics, Stability, divergence, dynamics_of, fixed_points, jacobian, nullclines, stability
from ._statistics import PowerLawFit, ccdf, fit_power_law, gini, hill, histogram
from ._symbolic import equations_of, solve_symbolically

__all__: list[str] = [
    'BasinStability',
    'Clusters',
    'Displacement',
    'Dynamics',
    'EarlyWarning',
    'Events',
    'NetworkSummary',
    'OrbitDiagram',
    'PowerLawFit',
    'Section',
    'Stability',
    'as_graph',
    'autocorrelation',
    'basin_stability',
    'block_entropies',
    'block_entropy',
    'box_counting_dimension',
    'ccdf',
    'centrality',
    'correlation_function',
    'correlation_information',
    'degrees',
    'detect_events',
    'divergence',
    'dynamics_of',
    'early_warning',
    'entropy_rate',
    'equations_of',
    'excess_entropy',
    'fit_power_law',
    'fixed_points',
    'gini',
    'hill',
    'histogram',
    'hurst_exponent',
    'jacobian',
    'kl_divergence',
    'label_clusters',
    'lempel_ziv_complexity',
    'lyapunov_spectrum',
    'mean_squared_displacement',
    'mutual_information',
    'nullclines',
    'orbit_diagram',
    'permutation_entropy',
    'poincare_section',
    'power_spectrum',
    'shannon_entropy',
    'solve_symbolically',
    'stability',
    'stroboscopic',
    'summarize_network',
    'symbolize',
    'transfer_entropy',
]
