# Tasks

## 1. Dependencies and errors

- [x] 1.1 Add `antropy`, `ordpy`, `nolds`, `powerlaw`, `ennemi`, `tidynamics`, `mrestimator`, `kramersmoyal`, and
  `deeptime` to the runtime dependencies of `pyproject.toml`, list them in the Project Profile,
  and verify a clean install with Python 3.12 and 3.13 imports each one, on macOS here and on Linux in CI
- [x] 1.2 Keep the errors of `skcomplex.core.base` without an extra error, widen the
  docstring of `AnalysisError` to every analysis, and verify the errors module tests pass

## 2. Layer 1: adapters, discovery, and signatures

- [x] 2.1 Add `DataKind`, `Series`, `JumpPath`, `Snapshots`, `Ensemble`, and the adapters `series_of`, `path_of`,
  `fields_of`, `snapshots_of`, `networks_of`, and `ensemble_of` with periodic unwrapping, and verify the series of an
  ODE, jump path, frames of a lattice, unwrapped agent positions, missing data kind, and replicates scenarios
- [x] 2.2 Add the `analysis` registration and `analyses_for` over trajectories, systems, experiment results, and
  arrays, register the existing functions with their data kinds, and verify the same function on two domains, arrays
  without a simulation, lattice model, agent model, and plain array scenarios
- [x] 2.3 Add `Signature` and `signature_analysis` for every data kind, add the gallery helper that plots a signature
  in `docs/gallery_tools`, and verify the signatures per data kind, model with several data kinds, and gallery figure
  scenarios
- [x] 2.4 Document data kinds, adapters, discovery, and signatures in the user guide with runnable examples, and verify
  the documentation test passes

## 3. Layer 2: existing measures moved onto libraries

- [x] 3.1 Replace the estimators of `permutation_entropy` and the ordinal coding of `symbolize` with ordpy, adding
  `delay`, and verify the long ordinal patterns scenario and equality with the old values at delay 1
- [x] 3.2 Replace `lempel_ziv_complexity` with antropy, and verify the Lempel-Ziv of a long sequence scenario and
  equality with the old values on fixed sequences
- [x] 3.3 Replace `hurst_exponent` with nolds, adding `method` and the fluctuation curve, and verify the white noise
  memory and persistent and integrated noise scenarios
- [x] 3.4 Replace the fit and comparisons of `fit_power_law` with powerlaw, keeping the bootstrap and `PowerLawFit`,
  and verify the recover an exponent, discrete samples, and truncated tail scenarios
- [x] 3.5 Replace `mean_squared_displacement` with tidynamics, adding the lag range and the diffusion coefficient, and
  extend `early_warning` with skewness and spectral reddening, and verify the Brownian motion and same early warning on
  two tipping points scenarios
- [x] 3.6 Update the docstrings and the user guide for the replaced estimators, report the changelog entry for the
  maintainer who writes the changelog, and verify the doctests and the documentation test pass

## 4. Layer 2: information and complexity

- [x] 4.1 Add Rényi and Tsallis entropies and the ensemble divergence to a reference law, and verify the uniform
  distribution, Tsallis limit, and H-theorem of a relaxing ensemble scenarios
- [x] 4.2 Add sample, approximate, multiscale, and spectral entropies through antropy, and verify the white noise,
  periodic orbits on two domains, multiscale signature, and spectral entropy scenarios
- [x] 4.3 Add two-dimensional permutation entropy and the complexity-entropy plane with its bounds through ordpy, and
  verify the complexity-entropy plane and fields of two domains scenarios
- [x] 4.4 Add active information storage and the nearest-neighbour transfer entropy through ennemi, and verify the
  periodic storage, linear Gaussian transfer, directed coupling of two domains, and direction of information flow
  scenarios
- [x] 4.5 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 5. Layer 2: dynamical invariants

- [x] 5.1 Add delay embedding with the mutual-information delay and false nearest neighbours, and verify the Lorenz
  embedding and map embedding scenarios
- [x] 5.2 Add the correlation dimension from `cKDTree` correlation sums, and verify the Lorenz attractor, Hénon
  attractor, and agents in a plane scenarios
- [x] 5.3 Add the Kaplan-Yorke dimension, the Kolmogorov-Sinai entropy, and the Rosenstein largest exponent through
  nolds, and verify the Kolmogorov-Sinai entropy of a map and a flow, Kaplan-Yorke dimension, and largest exponent from
  data scenarios
- [x] 5.4 Add invariant densities and Ulam's method, and verify the logistic map at r = 4 scenario
- [x] 5.5 Add recurrence matrices and recurrence quantification, and verify the order and noise and intermittency on two
  domains scenarios
- [x] 5.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 6. Layer 2: stochastic analysis

- [x] 6.1 Add first-passage times, return times, laminar durations, and hitting probabilities, and verify the drifted
  diffusion and gambler's ruin, return times on two domains, and fixation of a mutant scenarios
- [x] 6.2 Add stationary distributions, spectral gaps, relaxation times, transition estimates, and the detailed balance
  test, and verify the Ehrenfest urn and random walk on a ring and detailed balance scenarios
- [x] 6.3 Add Fano factors, relative fluctuations, and Kramers-Moyal coefficients through kramersmoyal, and verify the
  Poisson and bursty production, Ornstein-Uhlenbeck coefficients, and relative fluctuations scenarios
- [x] 6.4 Add entropy production from jump and Langevin laws and from transition counts, and the fluctuation theorem
  check, and verify the same entropy production on two domains, estimate from counts, and fluctuation theorem
  scenarios
- [x] 6.5 Add scaled cumulant generating functions from tilted generators and blocks, rate functions by
  Legendre-Fenchel transform and from histograms, and exponential reweighting, and verify the Poisson counting,
  Gallavotti-Cohen symmetry, mean-field order parameter, and reweighted magnetization scenarios
- [x] 6.6 Add Markov state models through deeptime as a runtime dependency, and verify the known three-state chain,
  double well on two domains, and default install scenarios
- [x] 6.7 Add two-time correlations, aging tests, ratio collapse, responses, and fluctuation-dissipation ratios, and
  verify the equilibrium has no aging, aging on two domains, and trap model distribution scenarios
- [x] 6.8 Add extreme-value fits, record counts, stable fits, and goodness-of-fit comparison with a named law, and
  verify the extreme-value classes, records of independent values, and Boltzmann law on two domains scenarios
- [x] 6.9 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 7. Layer 2: field analysis

- [x] 7.1 Add structure factors and correlation lengths by second moment and by fit, and verify the Ising lattice above
  its critical temperature and same correlation length on a reaction-diffusion field scenarios
- [x] 7.2 Add domain sizes and their growth exponent, and verify the two coarsening laws scenario
- [x] 7.3 Add pattern wavelengths, orientation order, and giant number fluctuations, and verify the Turing patterns on
  two domains and giant number fluctuations scenarios
- [x] 7.4 Add cluster size distributions of grids and graph components, percolation strength, and spanning
  probability, and verify the cluster sizes at the threshold on two kinds of data, forest fire clusters, and
  percolation threshold scenarios
- [x] 7.5 Add Binder cumulants, their crossings, and finite-size scaling collapse with its quality, and verify the
  Ising critical point and same collapse on two transitions scenarios
- [x] 7.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 8. Layer 2: agent analysis

- [x] 8.1 Add polar, nematic, milling, and phase order parameters, and verify the Vicsek ordering, same order parameter
  on oscillators and headings, and milling scenarios
- [x] 8.2 Add the kinetic temperature and the Maxwell-Boltzmann comparison in any dimension, and verify the
  Maxwell-Boltzmann in two and three dimensions and Langevin particles scenarios
- [x] 8.3 Add nearest-neighbour distributions, the Clark-Evans ratio, radial distribution functions, and coarse-grained
  density and polarization fields, and verify the random and ordered positions, radial distribution of a fluid, and
  coarse-grained density scenarios
- [x] 8.4 Wire the unwrapped agent positions into the displacement analysis, and verify the diffusion on two kinds of
  space and active particles scenarios
- [x] 8.5 Add the dissimilarity index, the share of like neighbours, and the interface density, and verify the same
  segregation index on two domains scenario
- [x] 8.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 9. Layer 2: criticality analysis

- [x] 9.1 Add avalanche extraction from activity series and relaxation records with sizes, durations, and shapes, and
  the fits of their laws, and verify the same avalanche statistics on a sandpile and an earthquake model and mean-field
  branching scenarios
- [x] 9.2 Add the crackling-noise scaling check and shape collapse, and verify the branching process scaling scenario
- [x] 9.3 Add the branching ratio through mrestimator, and verify the subsampled branching network and neuronal and
  epidemic activity scenarios
- [x] 9.4 Add order parameter sweeps, replica overlaps, susceptibilities, peaks, and exponent fits, and verify the same
  susceptibility peak on two transitions, order-parameter exponent, and replica overlap scenarios
- [x] 9.5 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 10. Layer 2: network analysis

- [x] 10.1 Add centralities, degree moments, clustering, and mean path length, and verify the small world and star
  centralities scenarios
- [x] 10.2 Add degree and attribute assortativity and the mean neighbour degree, and verify the attribute mixing on two
  domains scenario
- [x] 10.3 Add community detection with modularity and agreement with a reference partition, and verify the planted
  partition scenario
- [x] 10.4 Add motif counts of 3 and 4 nodes and their z-scores against seeded configuration-model and edge-switching
  ensembles, and verify the feed-forward loop scenario
- [x] 10.5 Add network measures over the frames of evolving networks, and verify the growing network and adaptive
  rewiring scenarios
- [x] 10.6 Add `network_structure_analysis` under `mathematics/networks` with its reporters, Rerun view, docstring
  example, inventory entry set to implemented, and gallery example, and verify the measures of a generated network,
  given graph, and gallery example scenarios and the inventory check
- [x] 10.7 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 11. Gate

- [ ] 11.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
