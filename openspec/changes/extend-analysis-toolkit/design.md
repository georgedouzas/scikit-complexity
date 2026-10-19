# Design

## Context

See proposal.md for the motivation.

`skcomplex.core.analysis` holds 13 private modules. The measures take arrays, but nothing connects them to the data a
model produces, so each example finds the right array of a `Trajectory` by hand:
- `Trajectory.__getitem__` returns the series of a state, a parameter, a reporter, or an observable.
- `Trajectory.lattice` returns lattice frames in grid shape.
- `Trajectory.agents` returns one `AgentSnapshot` of a group at a frame, with ids, values, and locations.
- `Trajectory.events` holds event and relaxation records.
- `ExperimentResult.series` stacks a named series over runs.

Some existing estimators repeat libraries. Some also have gaps:
- `lempel_ziv_complexity` is a pure-Python loop over list slices, so its cost grows with the cube of the length.
- `symbolize(..., 'ordinal')` codes a rank pattern by joining its digits, so patterns longer than 10 collide.
- `mean_squared_displacement` reads wrapped positions, takes lags in a direct loop, and fits all lags at once.
- `hurst_exponent` returns the detrended fluctuation exponent and offers no other estimator.

The archived `add-analysis-toolkit` change built these functions. `add-stochastic-maps` left Markov model inference to
a later analysis change, and `add-model-gallery` asks each example for an analysis figure from its own run.

The Bertin 2021 survey (key `Bt`) lists the methods the book applies to model outputs. This design takes every one of
them that reads data, and leaves out the ones that belong to a model:
- **Taken:** ensemble moments and relative fluctuations, stationary histograms against analytic laws, tail fits,
  first-return, laminar, and fixation statistics, `C(r)` with `xi`, two-time correlations, and giant number
  fluctuations.
- **Taken:** order parameters, replica overlaps, susceptibilities, and exponents, rate functions from histograms, the
  scaled cumulant generating function with its Legendre-Fenchel transform, and exponential reweighting.
- **Taken:** detailed balance, rate-matrix spectra, the time-dependent entropy with the divergence to the stationary
  law, aging collapse of `P(tau, t)`, and Kramers-Moyal coefficients.
- **Taken:** Ulam's method, invariant densities, coarse-graining agents to fields, extremes, records, stable laws, and
  clustering with path length.
- **Left out:** transfer matrices, real-space renormalization, the common-tangent construction, Itô-Stratonovich
  conversion, self-consistent mean-field equations, and Fisher's theorem. Each one reads a model's equations, not its
  output, and the symbolic analysis layer owns that.

## Goals / Non-Goals

**Goals:**

- One function per measure, keyed by data kind, never by domain.
- A thin adapter layer from `Trajectory` and `ExperimentResult` to plain arrays, so every measure also works on measured
  data.
- A default analysis per data kind that the 291 gallery examples can rely on.
- A library under every estimator that a mature library ships, with the in-house code limited to adapters and to
  measures no fitting library provides.

**Non-Goals:**

- Plotting. Signatures return curves, labels, and values. The gallery tools draw them with Matplotlib.
- Machine-learning classifiers of tipping points, such as the deep-learning model of `ewstools`.
- Topological data analysis, such as persistent homology. No permissive library has wheels for Python 3.13 on every
  platform the project supports.
- Convergent cross mapping. `pyEDM` carries a licence that is not OSI-approved.
- Model-side methods from the Bertin survey listed under Context.

## Decisions

### Data kinds and adapters

`DataKind` is an enumeration of `SERIES`, `PATH`, `FIELD`, `AGENTS`, `NETWORK`, and `ENSEMBLE`. The module `_adapters`
holds one function per kind:
- `series_of(trajectory, names=None)` returns a `Series` of times, values with one column per name, and labels. It reads
  states, reporters, and observables through `Trajectory.__getitem__`.
- `path_of(trajectory, name)` returns a `JumpPath` of jump times and states. It drops repeated frames of a discrete
  state, so it serves Gillespie runs, Boolean networks, and Markov chains alike.
- `fields_of(trajectory, name)` returns frames in grid shape for lattices, and for PDE fields through the field's grid.
- `snapshots_of(trajectory, group)` returns `Snapshots` of positions, velocities, headings, and states per frame. It
  also returns the box, and it unwraps periodic positions by accumulating minimum-image steps.
- `networks_of(trajectory)` returns one NetworkX graph per frame from the network snapshots of `add-adaptive-networks`.
- `ensemble_of(result, name)` returns an `Ensemble` of runs by times on common times, with the design rows and seeds.

Each analysis function takes the plain arrays. It also carries an `analysis` registration, a decorator that records
its data kind, its adapter, and its extra in one module-level table. `analyses_for(obj)` reads the problem's partition
forms, lattice groups, agent groups, networks, and observables, or the shape of an array, and returns the matching
table rows. The decorator keeps the fact of a function's data kind in one place, next to the function.

### Signature analyses

`signature_analysis(obj)` returns one `Signature` per data kind present, ordered agents, field, network, path, series,
ensemble. A `Signature` holds curves, axis labels, a title, and named values:
- **Series:** the Welch power spectrum on log axes with the integrated autocorrelation time. A map shows its return map
  instead, because its spectrum says little.
- **Path:** the complementary cumulative distribution of waiting times, with the Fano factor of the jump counts.
- **Field:** the radially averaged structure factor of the last frame. It states the peak wavelength when the spectrum
  has an interior peak, and the correlation length otherwise.
- **Agents:** the polar or milling order over time when velocities or headings exist, and the nearest-neighbour distance
  distribution against the Poisson law otherwise.
- **Network:** the degree distribution with its moments, clustering, and mean path length.
- **Ensemble:** the mean and the 5 to 95 percent band over runs.

The gallery tools of `add-model-gallery` plot a signature with one function, so an example that names no analysis
still meets the gallery requirement for an analysis figure from its own run.

### Libraries surveyed

PyPI metadata and GitHub were checked on 2026-10-19. "Wheels" means wheels for Linux and macOS on Python 3.12 and
3.13, or a pure-Python wheel.

- **SciPy** (1.18.1, 2026-08, BSD-3, wheels: yes). Used for welch spectra, FFT correlation, `cKDTree` with periodic
  `boxsize`, `ndimage.label`, sparse eigensolvers, `stats` fits. Core, already.
- **NetworkX** (3.7, 2026-09, BSD-3, wheels: pure). Used for centralities, assortativity, Louvain and label propagation,
  triadic census, edge swaps, configuration model. Core, already.
- **scikit-learn** (1.9.1, 2026-09, BSD-3, wheels: yes). Used for normalized mutual information of partitions. Core,
  already.
- **antropy** (0.2.3, 2026-10, BSD-3, wheels: pure). Used for sample, approximate, spectral entropy, Lempel-Ziv. Core,
  new, deps already present.
- **ordpy** (1.2.3, 2026-08, MIT, wheels: pure). Used for permutation entropy with delay and 2D patterns, Rényi and
  Tsallis forms, complexity-entropy plane and bounds. Core, new, no deps.
- **nolds** (0.6.3, 2025-11, MIT, wheels: pure). Used for detrended fluctuation, R/S Hurst, Rosenstein largest exponent.
  Core, new.
- **powerlaw** (2.0.0, 2026-01, MIT, wheels: pure). Used for power-law fits, exact discrete likelihood, truncated,
  stretched, and log-normal comparisons. Core, new, brings Matplotlib.
- **ennemi** (1.6.0, 2026-03, MIT, wheels: pure). Used for nearest-neighbour conditional mutual information for
  continuous transfer entropy. Core, new.
- **tidynamics** (1.1.2, 2023-07, BSD-3, wheels: pure). Used for FFT mean squared displacement and correlations over all
  lags. Core, new, stable small API.
- **mrestimator** (0.2.0, 2025-06, BSD-3, wheels: pure). Used for multistep regression branching ratio. Core, new.
- **kramersmoyal** (0.4.1, 2023-08, MIT, wheels: pure). Used for kernel Kramers-Moyal coefficients. Core, new.
- **deeptime** (0.4.5, 2025-04, LGPL-3.0, wheels: yes). Used for markov state models, implied timescales, PCCA+,
  committors, mean first passage. Extra `markov`.
- **EntropyHub** (2.0, 2024-04, Apache-2.0, wheels: pure). Used for multiscale entropy. Rejected because it has no
  release since 2024-04, pulls `requests`, `EMD-signal`, Matplotlib. Coarse-graining with antropy gives the same
  estimator.
- **neurokit2** (0.2.13, 2026-03, MIT, wheels: pure). Used for delay and false nearest neighbours. Rejected because
  pulls pandas, PyWavelets, `requests` for two functions, and duplicates antropy and nolds.
- **nolitsa** (GitHub only, BSD-3, wheels: none). Used for delay embedding. Rejected because it is not on PyPI.
- **driada** (1.2.1, 2026-10, MIT, wheels: pure). Used for false nearest neighbours. Rejected because it pulls `gdown`,
  GitPython, BeautifulSoup.
- **teaspoon** (1.6.0, 2026-06, GPL-3.0, wheels: pure). Used for embedding parameters. Rejected because it is copyleft,
  pulls gudhi, ripser, and pins pyparsing.
- **giotto-tda** (0.6.2, 2024-05, AGPL-3.0, wheels: no 3.13). Used for takens embedding search. Rejected because it is
  AGPL and no Python 3.13 wheels.
- **PyRQA** (8.1.0, 2024-02, Apache-2.0, wheels: sdist). Used for recurrence quantification. Rejected because it needs
  an OpenCL runtime, beta, sdist only.
- **pyunicorn** (1.0.0, 2026-07, BSD-3, wheels: sdist). Used for recurrence plots and networks. Rejected because sdist
  only with a Cython build, pulls igraph and h5netcdf.
- **SMdRQA, RECLAC** (2025-07, 2021-09, GPL-3.0, MIT, wheels: pure). Used for sliding-window RQA, recurrence lacunarity.
  Rejected because it is copyleft, and stale.
- **IDTxl** (GitHub only, GPL-3.0, wheels: none). Used for transfer entropy. Rejected because it is copyleft, not on
  PyPI, needs a Java runtime.
- **pyinform** (0.2.0, 2019-08, MIT, wheels: pure). Used for discrete information dynamics. Rejected because
  unmaintained since 2019.
- **infomeasure** (0.6.3, 2026-08, AGPL-3.0, wheels: pure). Used for information measures. Rejected because it is AGPL.
- **dit** (2.3, 2026-09, BSD-3, wheels: pure). Used for Rényi and Tsallis of distributions. Rejected because it is built
  for exact joint distributions, pulls boltons, debtcollector, loguru. The two entropies are closed forms over the
  probabilities.
- **pyitlib** (0.3.1, 2025-05, MIT, wheels: sdist). Used for discrete information measures. Rejected because it ships an
  sdist only, pulls pandas.
- **ewstools** (2.1.3, 2026-07, MIT, wheels: pure). Used for early-warning indicators. Rejected because it runs pandas
  rolling windows over a stack of plotly, arch, lmfit, EntropyHub. The indicators are NumPy windows and SciPy's Kendall
  tau.
- **statsmodels** (0.15.0, 2026-08, BSD-3, wheels: yes). Used for autocorrelation, LOWESS. Rejected because SciPy covers
  the correlations and spectra used here.
- **freud** (3.6.1, 2026-09, BSD-3, wheels: abi3, no macOS x86_64). Used for MSD, RDF, neighbour queries. Rejected
  because no wheel for macOS on Intel, which LAMMPS and the rest of the core support. SciPy's periodic `cKDTree` covers
  the queries.
- **MDAnalysis** (2.10.0, 2025-10, LGPL-3.0, wheels: yes). Used for MSD. Rejected because it is copyleft, built around
  molecular topologies.
- **fathon, MFDFA** (1.4.0, 0.4.3, GPL-3.0, MIT, wheels: yes, pure). Used for detrended fluctuation. Rejected because
  fathon is copyleft and MFDFA is stale since 2022. nolds covers DFA.
- **fssa** (0.7.6, 2015-12, ISC, wheels: pure). Used for finite-size scaling collapse. Rejected because it is stale
  since 2015, imports `future` for Python 2.
- **segregation** (2.6.0, 2026-09, BSD-3, wheels: pure). Used for segregation indices. Rejected because it is built on
  GeoPandas tables. The indices are counts over neighbours.
- **porespy** (3.1.1, 2026-09, MIT, wheels: pure). Used for box counting. Rejected because it pulls dask, edt, and
  Matplotlib inline for one function.
- **python-igraph** (n/a, GPL-2.0, wheels: yes). Used for motif census. Rejected because it is copyleft. NetworkX counts
  the motifs of 3 and 4 nodes.
- **pyEDM** (2.5.7, 2026-08, UCSD licence, wheels: pure). Used for convergent cross mapping. Rejected because its
  licence not OSI-approved.
- **PyEMMA, msmtools, PyDTMC** (2022, 2021, 2026, LGPL-3.0, LGPL-3.0, MIT, wheels: sdist, sdist, pure). Used for markov
  models. Rejected because PyEMMA and msmtools are retired in favour of deeptime. PyDTMC was rejected by
  `add-stochastic-maps`.

### Measures without a fitting library

These measures are built in house on NumPy and SciPy. Each one records why no library fits:
- **Delay and false nearest neighbours:** reuse `mutual_information` on quantile symbols for the delay, and run Kennel's
  test with SciPy's `cKDTree`. The candidates above are rejected.
- **Correlation dimension:** correlation sums from `cKDTree.count_neighbors` over radii, in `O(N log N)`.
  `nolds.corr_dim` builds the full distance matrix, which needs 3 GB for 20,000 points.
- **Recurrence quantification:** sparse recurrence matrices from `cKDTree.sparse_distance_matrix`, and line lengths from
  the runs of the diagonals and the columns. Every RQA library is rejected above.
- **Kaplan-Yorke dimension, Kolmogorov-Sinai entropy, Ulam's method, and invariant densities:** short formulas over the
  existing `lyapunov_spectrum` and NumPy histograms.
- **Entropy production, fluctuation theorem, tilted generators, and Legendre-Fenchel transforms:** no Python library
  exists. The tilted generator's largest eigenvalue comes from `scipy.linalg.eigvals`, and the transform is a maximum
  over a grid.
- **Aging and fluctuation-dissipation ratios:** NumPy means over the ensemble axis.
- **Structure factors, domain sizes, wavelengths, Binder cumulants, and collapse:** `scipy.fft` and the existing
  `correlation_function`, with the collapse quality of Houdayer and Hartmann minimized by `scipy.optimize`.
- **Order parameters, segregation indices, and Clark-Evans:** closed forms over arrays and `cKDTree` neighbours.
- **Avalanche shapes and crackling-noise scaling:** `detect_events`, log-binned means, and `powerlaw` fits.
- **Extremes, records, and stable laws:** `scipy.stats.genextreme`, `scipy.stats.levy_stable`, and `scipy.stats`
  goodness-of-fit tests.
- **Motif z-scores:** NetworkX's triadic census and graph matching for 4-node motifs, against
  `networkx.directed_edge_swap`, `double_edge_swap`, or `configuration_model`, seeded from the caller.

### Audit of existing functions

- `permutation_entropy`: the current estimator is ordinal symbols, Shannon entropy, no delay. Library: ordpy. Verdict:
  Replace. Adds `delay`, 2D fields, and Rényi and Tsallis forms. Same value at delay 1.
- `symbolize(..., 'ordinal')`: the current estimator is digit-joined rank codes, which collide above length 10. Library:
  ordpy patterns. Verdict: Replace the coding with a lexicographic rank index, and keep the signature.
- `lempel_ziv_complexity`: the current estimator is a pure-Python LZ76 with list slices. Library: antropy
  `lziv_complexity`. Verdict: Replace. Same normalization `n / log_k n`, Numba speed.
- `hurst_exponent`: the current estimator is an in-house DFA, correct. Library: nolds `dfa`, `hurst_rs`. Verdict:
  Replace. Adds `method='dfa'` or `'rs'`, overlapping windows, and the fluctuation curve. Document that DFA gives `H +
  1` on integrated series.
- `fit_power_law`: the current estimator is Clauset, Shalizi, and Newman with the exact discrete likelihood, correct.
  Library: powerlaw. Verdict: Replace the fit and the comparisons with `powerlaw.Fit`, which adds `xmax`, truncated and
  stretched tails. Keep the bootstrap p-value in the adapter, which powerlaw does not give. `PowerLawFit` keeps its
  fields and gains `comparisons` entries.
- `mean_squared_displacement`: the current estimator is a direct loop over 20 lags, wrapped positions, fit over all
  lags. Library: tidynamics `msd`. Verdict: Replace. FFT over all lags, unwrapped positions from the adapter, exponent
  over a chosen lag range.
- `transfer_entropy`: the current estimator is plug-in counts on symbols, correct. Library: none permissive and
  maintained. Verdict: Keep, and add `transfer_entropy(..., estimator='knn')` through ennemi for continuous values.
- `box_counting_dimension`: the current estimator is NumPy reshape counts, correct. Library: porespy rejected. Verdict:
  Keep.
- `autocorrelation`, `power_spectrum`: the current estimator is NumPy FFT, SciPy Welch. Library: SciPy. Verdict: Keep.
- `early_warning`: the current estimator is NumPy windows, Kendall tau. Library: ewstools rejected. Verdict: Keep, and
  add skewness and spectral reddening.
- `label_clusters`, `correlation_function`: the current estimator is `scipy.ndimage.label`, FFT. Library: SciPy.
  Verdict: Keep.
- Shannon, block, rate, excess, mutual information: the current estimator is plug-in counts. Library: dit rejected.
  Verdict: Keep.
- `lyapunov_spectrum`: the current estimator is tangent dynamics of the compiled system. Library: no library reads
  compiled problems. Verdict: Keep, and add the Rosenstein estimate from data through nolds.
- `ccdf`, `histogram`, `hill`, `gini`: the current estimator is NumPy. Library: none needed. Verdict: Keep.

Every replaced function keeps its public name, parameters, and result type. Its tests now compare it with the old
in-house values on fixed seeds within 1e-8, or within the stated tolerance where the library's estimator differs.

### Optional extra and errors

`pyproject.toml` declares `markov = ["deeptime>=0.4.5"]`. `_markov` imports `deeptime` inside its functions, the one
lazy import the constitution allows, with a suppression that names the extra. A failed import raises
`MissingExtraError`. This change adds it to `skcomplex.core.base` as a `BackendError` if `add-constraint-based-models`
has not added it first. Transition-matrix analysis of a given matrix needs no extra and uses SciPy.

Analysis errors raise `AnalysisError`. Its docstring changes from "a symbolic analysis" to "an analysis", because
every family now raises it.

### Module layout

The new modules are `_adapters`, `_signatures`, `_complexity`, `_embedding`, `_invariants`, `_recurrence`, `_passage`,
`_markov`, `_irreversibility`, `_large_deviations`, `_aging`, `_extremes`, `_fields`, `_scaling`, `_agents`,
`_criticality`, and `_motifs`. Each holds one concern, and all re-export from `skcomplex.core.analysis`. The model
`network_structure_analysis` lives in `skcomplex.models.mathematics.networks`, reports through model reporters, and
uses only the public analysis surface.

## Risks / Trade-offs

- [`powerlaw` and `mrestimator` pull Matplotlib into the default install] → Matplotlib already serves the docs and the
  gallery. The core imports neither library's plotting code.
- [tidynamics and kramersmoyal have had no release since 2023] → Both are small NumPy-only modules with a stable API.
  The tests pin their results, so a regression shows up at once.
- [Replacing estimators shifts results in the last digits] → The audit table states which functions change. The
  changelog lists them, and the tests compare old and new values on fixed seeds.
- [Cross-domain scenarios need library models that other changes still add, such as OFC, Vicsek, and Schelling] → Each
  test builds the model from the library when it exists, and from a minimal system in the test module otherwise, so no
  task blocks on another change.
- [Scenarios with physical constants are slow, such as Ising near criticality] → They run under the `slow` marker in the
  nightly test session, with smaller sizes and wider tolerances in the default session.

## Migration Plan

No public name is removed. Replaced estimators keep their signatures, and new keyword arguments have defaults that
reproduce the old behaviour. Rolling back means reverting the change and dropping the new dependencies and the extra.
