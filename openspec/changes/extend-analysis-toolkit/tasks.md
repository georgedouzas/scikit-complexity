# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `antropy`, `ordpy`, `nolds`, `powerlaw`, `ennemi`, `tidynamics`, `mrestimator`, and `kramersmoyal` to the
  runtime dependencies and the `markov` extra with `deeptime` to `pyproject.toml`, list them in the Project Profile,
  and verify a clean install on Linux and macOS with Python 3.12 and 3.13 imports each one
- [ ] 1.2 Add `MissingExtraError` to `skcomplex.core.base` unless `add-constraint-based-models` has added it, widen the
  docstring of `AnalysisError` to every analysis, and verify the errors module tests pass

## 2. Layer 1: adapters, discovery, and signatures

- [ ] 2.1 Add `DataKind`, `Series`, `JumpPath`, `Snapshots`, `Ensemble`, and the adapters `series_of`, `path_of`,
  `fields_of`, `snapshots_of`, `networks_of`, and `ensemble_of` with periodic unwrapping, and verify the series of an
  ODE, jump path, frames of a lattice, unwrapped agent positions, missing data kind, and replicates scenarios
- [ ] 2.2 Add the `analysis` registration and `analyses_for` over trajectories, systems, experiment results, and
  arrays, register the existing functions with their data kinds, and verify the same function on two domains, arrays
  without a simulation, lattice model, agent model, and plain array scenarios
- [ ] 2.3 Add `Signature` and `signature_analysis` for every data kind, add the gallery helper that plots a signature
  in `docs/gallery_tools`, and verify the signatures per data kind, model with several data kinds, and gallery figure
  scenarios
- [ ] 2.4 Document data kinds, adapters, discovery, and signatures in the user guide with runnable examples, and verify
  the documentation test passes

## 3. Layer 2: existing measures moved onto libraries

- [ ] 3.1 Replace the estimators of `permutation_entropy` and the ordinal coding of `symbolize` with ordpy, adding
  `delay`, and verify the long ordinal patterns scenario and equality with the old values at delay 1
- [ ] 3.2 Replace `lempel_ziv_complexity` with antropy, and verify the Lempel-Ziv of a long sequence scenario and
  equality with the old values on fixed sequences
- [ ] 3.3 Replace `hurst_exponent` with nolds, adding `method` and the fluctuation curve, and verify the white noise
  memory and persistent and integrated noise scenarios
- [ ] 3.4 Replace the fit and comparisons of `fit_power_law` with powerlaw, keeping the bootstrap and `PowerLawFit`,
  and verify the recover an exponent, discrete samples, and truncated tail scenarios
- [ ] 3.5 Replace `mean_squared_displacement` with tidynamics, adding the lag range and the diffusion coefficient, and
  extend `early_warning` with skewness and spectral reddening, and verify the Brownian motion and same early warning on
  two tipping points scenarios
- [ ] 3.6 Update the docstrings, the user guide, and the changelog for the replaced estimators, and verify the doctests
  and the documentation test pass

## 4. Layer 2: information and complexity

- [ ] 4.1 Add Rényi and Tsallis entropies and the ensemble divergence to a reference law, and verify the uniform
  distribution, Tsallis limit, and H-theorem of a relaxing ensemble scenarios
- [ ] 4.2 Add sample, approximate, multiscale, and spectral entropies through antropy, and verify the white noise,
  periodic orbits on two domains, multiscale signature, and spectral entropy scenarios
- [ ] 4.3 Add two-dimensional permutation entropy and the complexity-entropy plane with its bounds through ordpy, and
  verify the complexity-entropy plane and fields of two domains scenarios
- [ ] 4.4 Add active information storage and the nearest-neighbour transfer entropy through ennemi, and verify the
  periodic storage, linear Gaussian transfer, directed coupling of two domains, and direction of information flow
  scenarios
- [ ] 4.5 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 5. Layer 2: dynamical invariants

- [ ] 5.1 Add delay embedding with the mutual-information delay and false nearest neighbours, and verify the Lorenz
  embedding and map embedding scenarios
- [ ] 5.2 Add the correlation dimension from `cKDTree` correlation sums, and verify the Lorenz attractor, Hénon
  attractor, and agents in a plane scenarios
- [ ] 5.3 Add the Kaplan-Yorke dimension, the Kolmogorov-Sinai entropy, and the Rosenstein largest exponent through
  nolds, and verify the Kolmogorov-Sinai entropy of a map and a flow, Kaplan-Yorke dimension, and largest exponent from
  data scenarios
- [ ] 5.4 Add invariant densities and Ulam's method, and verify the logistic map at r = 4 scenario
- [ ] 5.5 Add recurrence matrices and recurrence quantification, and verify the order and noise and intermittency on two
  domains scenarios
- [ ] 5.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 6. Layer 2: stochastic analysis

- [ ] 6.1 Add first-passage times, return times, laminar durations, and hitting probabilities, and verify the drifted
  diffusion and gambler's ruin, return times on two domains, and fixation of a mutant scenarios
- [ ] 6.2 Add stationary distributions, spectral gaps, relaxation times, transition estimates, and the detailed balance
  test, and verify the Ehrenfest urn and random walk on a ring and detailed balance scenarios
- [ ] 6.3 Add Fano factors, relative fluctuations, and Kramers-Moyal coefficients through kramersmoyal, and verify the
  Poisson and bursty production, Ornstein-Uhlenbeck coefficients, and relative fluctuations scenarios
- [ ] 6.4 Add entropy production from jump and Langevin laws and from transition counts, and the fluctuation theorem
  check, and verify the same entropy production on two domains, estimate from counts, and fluctuation theorem
  scenarios
- [ ] 6.5 Add scaled cumulant generating functions from tilted generators and blocks, rate functions by
  Legendre-Fenchel transform and from histograms, and exponential reweighting, and verify the Poisson counting,
  Gallavotti-Cohen symmetry, mean-field order parameter, and reweighted magnetization scenarios
- [ ] 6.6 Add Markov state models through deeptime behind the `markov` extra, and verify the known three-state chain,
  double well on two domains, Markov state model without the extra, and default install scenarios
- [ ] 6.7 Add two-time correlations, aging tests, ratio collapse, responses, and fluctuation-dissipation ratios, and
  verify the equilibrium has no aging, aging on two domains, and trap model distribution scenarios
- [ ] 6.8 Add extreme-value fits, record counts, stable fits, and goodness-of-fit comparison with a named law, and
  verify the extreme-value classes, records of independent values, and Boltzmann law on two domains scenarios
- [ ] 6.9 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 7. Layer 2: field analysis

- [ ] 7.1 Add structure factors and correlation lengths by second moment and by fit, and verify the Ising lattice above
  its critical temperature and same correlation length on a reaction-diffusion field scenarios
- [ ] 7.2 Add domain sizes and their growth exponent, and verify the two coarsening laws scenario
- [ ] 7.3 Add pattern wavelengths, orientation order, and giant number fluctuations, and verify the Turing patterns on
  two domains and giant number fluctuations scenarios
- [ ] 7.4 Add cluster size distributions of grids and graph components, percolation strength, and spanning
  probability, and verify the cluster sizes at the threshold on two kinds of data, forest fire clusters, and
  percolation threshold scenarios
- [ ] 7.5 Add Binder cumulants, their crossings, and finite-size scaling collapse with its quality, and verify the
  Ising critical point and same collapse on two transitions scenarios
- [ ] 7.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 8. Layer 2: agent analysis

- [ ] 8.1 Add polar, nematic, milling, and phase order parameters, and verify the Vicsek ordering, same order parameter
  on oscillators and headings, and milling scenarios
- [ ] 8.2 Add the kinetic temperature and the Maxwell-Boltzmann comparison in any dimension, and verify the
  Maxwell-Boltzmann in two and three dimensions and Langevin particles scenarios
- [ ] 8.3 Add nearest-neighbour distributions, the Clark-Evans ratio, radial distribution functions, and coarse-grained
  density and polarization fields, and verify the random and ordered positions, radial distribution of a fluid, and
  coarse-grained density scenarios
- [ ] 8.4 Wire the unwrapped agent positions into the displacement analysis, and verify the diffusion on two kinds of
  space and active particles scenarios
- [ ] 8.5 Add the dissimilarity index, the share of like neighbours, and the interface density, and verify the same
  segregation index on two domains scenario
- [ ] 8.6 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 9. Layer 2: criticality analysis

- [ ] 9.1 Add avalanche extraction from activity series and relaxation records with sizes, durations, and shapes, and
  the fits of their laws, and verify the same avalanche statistics on a sandpile and an earthquake model and mean-field
  branching scenarios
- [ ] 9.2 Add the crackling-noise scaling check and shape collapse, and verify the branching process scaling scenario
- [ ] 9.3 Add the branching ratio through mrestimator, and verify the subsampled branching network and neuronal and
  epidemic activity scenarios
- [ ] 9.4 Add order parameter sweeps, replica overlaps, susceptibilities, peaks, and exponent fits, and verify the same
  susceptibility peak on two transitions, order-parameter exponent, and replica overlap scenarios
- [ ] 9.5 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 10. Layer 2: network analysis

- [ ] 10.1 Add centralities, degree moments, clustering, and mean path length, and verify the small world and star
  centralities scenarios
- [ ] 10.2 Add degree and attribute assortativity and the mean neighbour degree, and verify the attribute mixing on two
  domains scenario
- [ ] 10.3 Add community detection with modularity and agreement with a reference partition, and verify the planted
  partition scenario
- [ ] 10.4 Add motif counts of 3 and 4 nodes and their z-scores against seeded configuration-model and edge-switching
  ensembles, and verify the feed-forward loop scenario
- [ ] 10.5 Add network measures over the frames of evolving networks, and verify the growing network and adaptive
  rewiring scenarios
- [ ] 10.6 Add `network_structure_analysis` under `mathematics/networks` with its reporters, Rerun view, docstring
  example, inventory entry set to implemented, and gallery example, and verify the measures of a generated network,
  given graph, and gallery example scenarios and the inventory check
- [ ] 10.7 Document the family in the user guide with runnable examples, and verify the documentation test passes

## 11. Gate

- [ ] 11.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
