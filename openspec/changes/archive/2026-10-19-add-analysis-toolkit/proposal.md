# Proposal

## Why

The analysis layer offers only `equations_of` and `solve_symbolically`, while every reference measures complex systems
with a shared toolkit:
- **Stability and chaos:** fixed points, stability, Lyapunov exponents, and bifurcation diagrams (Datseris and Parlitz
  2022 ch. 1-4, Dilão 2023 ch. 1-4).
- **Heavy tails and avalanches:** power-law fits and avalanche statistics (Thurner et al. 2018 §3.4, Charbonneau 2017
  ch. 5 and app. B, Caiani et al. 2016 ch. 4).
- **Information:** entropies, entropy rates, excess entropy, mutual and transfer information (Lindgren 2024 ch. 2-4,
  Jensen 2023 ch. 9).
- **Networks:** network measures (Thurner et al. 2018 ch. 4, Gros 2015 ch. 1).
- **Time series:** diffusion and long-memory statistics, and early-warning signals of tipping points (Argun et al. 2021
  ch. 5-6, Jensen 2023 ch. 10, 13).

Jensen (2023 ch. 2) also turns his taxonomy of emergence into a checklist of measurable signatures. This change adds
that toolkit, field-agnostic, on trajectories and compiled problems.

## What Changes

- Add symbolic dynamics analysis: fixed points, Jacobians, linear stability classification, nullclines, and the
  divergence of the vector field.
- Add numeric dynamics analysis: Lyapunov exponent spectra from tangent dynamics, orbit and bifurcation diagrams over
  experiment sweeps, and basin stability.
- Add statistics of distributions: log-binned histograms, complementary cumulative distributions, power-law fits by
  maximum likelihood with goodness of fit, Hill estimators, and Gini coefficients.
- Add event statistics: detection of avalanches and bursts in series with their sizes, durations, and waiting times.
- Add time-series measures: autocorrelation, power spectra, mean squared displacement with its exponent, Hurst exponent
  by detrended fluctuation analysis, and early-warning indicators.
- Add information measures: Shannon entropy, Kullback-Leibler divergence, block entropies, entropy rate, excess
  entropy, mutual information, transfer entropy, permutation entropy, and Lempel-Ziv complexity, with symbolisation
  helpers.
- Add spatial and network measures: cluster labelling and size distributions, correlation functions, box-counting
  dimension, and network degree distributions, clustering, path lengths, components, and centralities.
- Add a `nonlinear` library module whose models double as reference cases for the analysis: Lorenz, Rössler, Hénon,
  logistic map, Duffing, Van der Pol, and Hénon-Heiles.

## Capabilities

### New Capabilities

- `trajectory-analysis`: numeric analysis of trajectories, series, distributions, fields, and networks.

### Modified Capabilities

- `symbolic-analysis`: fixed points, Jacobians, stability, and vector-field divergence.
- `model-library`: the nonlinear dynamics models.

## Impact

- `skcomplex.analysis` grows submodules `dynamics`, `statistics`, `events`, `series`, `information`, `spatial`, and
  `networks`, re-exported from `skcomplex.analysis`.
- Depends on `add-general-model-interface` for observables and on `add-experiments` for sweeps and basin sampling.
