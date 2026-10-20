# Proposal

## Why

`skcomplex.core.analysis` holds about 50 measures, but a user must know which array of a `Trajectory` each one wants,
and the measures cover only part of what the references use on complex systems. The analyses must be general enough to
apply in every domain. So they are grouped by the kind of data a core model type produces, never by scientific field.
The same `correlation_length` then serves an Ising lattice and a reaction-diffusion field, and the same
`avalanche_statistics` serves a sandpile and an earthquake model.

The references use these measures:
- **Information and complexity:** Rényi and Tsallis entropies, sample and multiscale entropy, the complexity-entropy
  plane, and information storage and transfer (Lindgren 2024 ch. 3-5, Thurner et al. 2018 ch. 6).
- **Invariants of chaos:** delay embedding, correlation dimension, Kaplan-Yorke dimension, Kolmogorov-Sinai entropy,
  and recurrence quantification (Datseris and Parlitz 2022 ch. 5-6).
- **Stochastic processes:** first passage, stationary distributions and spectral gaps, entropy production and
  fluctuation theorems, large deviations, aging, and response (Bertin 2021 ch. 2-4, Argun et al. 2021 ch. 5-6).
- **Fields and lattices:** structure factors, correlation lengths, domain growth, percolation, and finite-size scaling
  (Bertin 2021, Charbonneau 2017 ch. 3-5).
- **Agents and collective motion:** polarization, milling, neighbour statistics, and segregation (Vicsek and Zafeiris
  2012, Bertin 2021).
- **Networks:** centralities, mixing, communities, and motifs (Rm 3.1-3.4, Bertin 2021 ch. 6).
- **Criticality:** avalanches, branching ratios, crackling-noise scaling, and susceptibility peaks (Charbonneau 2017
  ch. 5, Pruessner 2012).

Several existing functions also re-implement estimators that mature libraries ship faster or with more options. The
constitution asks for the library in that case.

## What Changes

- Add trajectory adapters that extract each data kind from a `Trajectory` or an `ExperimentResult`: series, jump paths,
  fields, agent snapshots, networks, and ensembles. Every analysis takes plain arrays, so it works on any data.
- Add `analyses_for`, which lists the analyses that apply to a trajectory, a system, or an array with their data kind.
- Add `signature_analysis`, one default analysis per data kind, which the model gallery plots for every example.
- Add information and complexity measures: Rényi and Tsallis entropies, sample, approximate, multiscale, and spectral
  entropies, the complexity-entropy plane, continuous transfer entropy, and active information storage.
- Add dynamical invariants: delay embedding with the mutual-information delay and false nearest neighbours, correlation
  dimension, the largest Lyapunov exponent from data, Kaplan-Yorke dimension, Kolmogorov-Sinai entropy, and recurrence
  quantification analysis.
- Add stochastic analysis: first-passage times, stationary distributions and spectral gaps, Fano factors, entropy
  production with fluctuation-theorem checks, scaled cumulant generating functions and rate functions, Markov state
  models, two-time correlations and aging, response functions, and fluctuation-dissipation ratios.
- Add field analysis: structure factors, correlation lengths, domain growth, pattern wavelengths, cluster size
  distributions, percolation observables, Binder cumulants, and finite-size scaling collapse.
- Add agent analysis: polarization, milling, kinetic temperature and Boltzmann fits, nearest-neighbour statistics,
  radial distribution functions, and segregation indices.
- Add criticality analysis: avalanche statistics with power-law fits and model comparison, crackling-noise scaling,
  branching ratios, and susceptibility peaks.
- Add network analysis: centralities, assortativity, community detection, motif z-scores against configuration-model
  or edge-switching ensembles, and measures over the frames of evolving networks.
- Add the library model `network_structure_analysis` under `mathematics/networks`, which records a generated or given
  network and reports its structure.
- Extend the existing measures: detrended fluctuation with a chosen order and R/S Hurst estimates, unwrapped mean
  squared displacement over all lags, more early-warning indicators, and ordinal patterns with a delay and on fields.
- Replace the in-house estimators of `permutation_entropy`, `lempel_ziv_complexity`, `hurst_exponent`,
  `fit_power_law`, and `mean_squared_displacement` with library calls. Their public names and result types stay.

## Capabilities

### New Capabilities

- `analysis-adapters`: data kinds, trajectory adapters, the discovery of applicable analyses, and signature analyses per
  data kind.
- `complexity-measures`: entropies of distributions and series, multiscale and spectral entropies, the
  complexity-entropy plane, and information storage and transfer.
- `dynamical-invariants`: delay embedding, fractal dimensions, Lyapunov exponents from data, Kolmogorov-Sinai entropy,
  and recurrence quantification.
- `stochastic-analysis`: first passage, stationary statistics, counting statistics, entropy production, large
  deviations, Markov state models, aging, and response.
- `field-analysis`: structure factors, correlation lengths, coarsening, patterns, clusters, percolation, and
  finite-size scaling of lattices and fields.
- `agent-analysis`: order parameters, velocity and energy distributions, neighbour statistics, and segregation of
  agent snapshots.
- `criticality-analysis`: avalanches, crackling-noise scaling, branching ratios, order parameters, and
  susceptibility peaks.
- `network-analysis`: centralities, clustering and paths, mixing, communities, motifs against null ensembles, and
  measures of evolving networks.

### Modified Capabilities

- `trajectory-analysis`: power-law fits with more alternatives, detrended fluctuation and Hurst estimates, unwrapped
  displacement, early-warning indicators, and ordinal patterns with a delay.
- `model-library`: `network_structure_analysis` under `mathematics/networks`, which the inventory marks implemented.

## Impact

- `skcomplex.core.analysis` gains private modules `_kinds`, `_adapters`, `_signatures`, `_complexity`, `_embedding`,
  `_invariants`, `_recurrence`, `_passage`, `_transitions`, `_fluctuations`, `_markov`, `_irreversibility`,
  `_large_deviations`, `_aging`, `_extremes`, `_fields`, `_percolation`, `_scaling`, `_agents`, `_segregation`,
  `_criticality`, and `_motifs`, re-exported from `skcomplex.core.analysis`. `_networks` grows communities, mixing,
  and evolving networks.
- `skcomplex.models.mathematics.networks` gains `network_structure_analysis`, and its inventory entry is implemented.
- New runtime dependencies with Linux and macOS wheels for Python 3.12 and 3.13: `antropy`, `ordpy`, `nolds`,
  `powerlaw`, `ennemi`, `tidynamics`, `mrestimator`, `kramersmoyal`, and `deeptime`. `powerlaw` and `mrestimator`
  bring `matplotlib` into the default install. `deeptime` is LGPL-3.0, and the core uses it, so by the Project Profile
  rule on dependency placement it is a runtime dependency, not an optional extra.
- `skcomplex.core.base` keeps its errors. `AnalysisError` now covers every analysis, and no extra is needed, so no
  `MissingExtraError` is added.
- The gallery of `add-model-gallery` draws each example's analysis figure from `signature_analysis` unless the example
  names a more specific analysis.
- Relates to `add-stochastic-maps`, which leaves Markov model inference to a later analysis change, and to
  `add-group-cellular-automata`, which reuses the block entropies.
