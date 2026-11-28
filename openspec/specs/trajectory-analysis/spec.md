# trajectory-analysis Specification

## Purpose

Gives a user field-agnostic measures of complex dynamics on trajectories and data: chaos, distributions, events, time
series, information, spatial structure, and networks.

## Requirements

### Requirement: Lyapunov exponents measure chaos
The analysis layer SHALL compute the Lyapunov spectrum of continuous systems and maps from their tangent dynamics.

#### Scenario: Lorenz attractor
- **WHEN** a user computes the largest Lyapunov exponent of the Lorenz-63 system with σ = 10, ρ = 28, β = 8/3
- **THEN** the result lies within 0.05 of 0.906

#### Scenario: Logistic map
- **WHEN** a user computes the Lyapunov exponent of the logistic map at r = 4
- **THEN** the result lies within 0.01 of ln 2

### Requirement: Orbit diagrams and basins come from sweeps
The analysis layer SHALL build orbit diagrams over a parameter range and basin stability estimates over initial
conditions, from experiment sweeps.

#### Scenario: Period doubling
- **WHEN** a user builds the orbit diagram of the logistic map for r in [2.8, 4]
- **THEN** the diagram has one point per r below 3, two between 3 and 3.449, and four between 3.449 and 3.544

### Requirement: Heavy tails are fitted and tested
The analysis layer SHALL compute log-binned histograms, complementary cumulative distributions, maximum-likelihood
power-law fits of continuous or discrete samples with their lower cutoff, an optional upper cutoff, and bootstrap
goodness of fit, comparisons with exponential, log-normal, stretched exponential, and truncated power-law tails, Hill
estimates, and Gini coefficients.

#### Scenario: Recover an exponent
- **WHEN** a user fits 10,000 samples drawn from a power law with exponent 2.5 above 1
- **THEN** the fitted exponent lies within 0.1 of 2.5 and the goodness-of-fit p-value exceeds 0.1

#### Scenario: Discrete samples
- **WHEN** a user fits 10,000 integer samples drawn from a discrete power law with exponent 2.5 above 1
- **THEN** the fitted exponent lies within 0.1 of 2.5 with the exact discrete likelihood

#### Scenario: Truncated tail
- **WHEN** a user compares the fits of 10,000 samples of a power law with exponent 1.5 and an exponential cutoff at 100
- **THEN** the truncated power law is preferred over the pure power law with a p-value below 0.05

#### Scenario: Gini of equal wealth
- **WHEN** a user computes the Gini coefficient of equal values
- **THEN** the result is 0

### Requirement: Events are detected in series
The analysis layer SHALL detect events, runs of a series above a threshold or relaxations that happen between drive
steps, and report their sizes, durations, and waiting times.

#### Scenario: Avalanche sizes
- **WHEN** a user detects avalanches in a series of topplings per drive step
- **THEN** each avalanche's size equals the sum of topplings over its consecutive nonzero steps

### Requirement: Time-series measures characterise memory and diffusion
The analysis layer SHALL compute autocorrelation, power spectra, mean squared displacement over all lags with its
exponent over a chosen range, detrended fluctuation of any order, Hurst exponents by detrended fluctuation or rescaled
range, and rolling variance, autocorrelation, skewness, and spectral reddening as early-warning indicators.

#### Scenario: Brownian motion
- **WHEN** a user computes the mean squared displacement of 1000 simulated Brownian particles
- **THEN** the scaling exponent lies within 0.05 of 1

#### Scenario: White noise memory
- **WHEN** a user computes the Hurst exponent of white noise
- **THEN** the result lies within 0.05 of 0.5

#### Scenario: Persistent and integrated noise
- **WHEN** a user computes the detrended fluctuation exponent of fractional Gaussian noise with Hurst exponent 0.7 and
  of a random walk
- **THEN** the first lies within 0.05 of 0.7 and the second within 0.05 of 1.5

#### Scenario: Same early warning on two tipping points
- **WHEN** a user computes early-warning indicators of a noisy harvested population approaching its fold and of a noisy
  lake phosphorus system approaching its fold
- **THEN** in both the Kendall trends of variance and lag-1 autocorrelation exceed 0.5

### Requirement: Information measures quantify structure
The analysis layer SHALL compute Shannon entropy, Kullback-Leibler divergence, block entropies, entropy rate, excess
entropy, mutual information, transfer entropy, permutation entropy with an embedding delay, and Lempel-Ziv complexity,
from symbols or from series symbolised by bins, quantiles, or ordinal patterns of any length.

#### Scenario: Periodic sequence
- **WHEN** a user computes the entropy rate and excess entropy of the repeated sequence `0011`
- **THEN** the entropy rate is 0 and the excess entropy is 2 bits

#### Scenario: Direction of information flow
- **WHEN** a user computes transfer entropy between a random binary series `x` and `y` that copies `x` with a lag of one
- **THEN** the transfer entropy from `x` to `y` is close to 1 bit and from `y` to `x` close to 0

#### Scenario: Long ordinal patterns
- **WHEN** a user symbolises white noise with ordinal patterns of length 12
- **THEN** two distinct rank patterns never share a symbol

#### Scenario: Lempel-Ziv of a long sequence
- **WHEN** a user computes the normalized Lempel-Ziv complexity of 10^5 fair coin flips
- **THEN** the result lies within 0.1 of 1 and the call returns within one second

### Requirement: Spatial and network structure is measured
The analysis layer SHALL label clusters on grids with their size distribution, compute spatial correlation functions
and box-counting dimensions, and compute degree distributions, clustering, path lengths, components, and centralities
of network snapshots.

#### Scenario: Percolation threshold
- **WHEN** a user labels clusters of site percolation on a 200 × 200 grid at occupation probabilities around 0.5927
- **THEN** the probability of a spanning cluster crosses one half within 0.02 of 0.5927

#### Scenario: Sierpinski dimension
- **WHEN** a user computes the box-counting dimension of a Sierpinski triangle image
- **THEN** the result lies within 0.05 of log 3 / log 2

### Requirement: Repeated Lyapunov spectra reuse their compiled tangent dynamics
`lyapunov_spectrum` SHALL compile the tangent dynamics of a system structure once, with the parameters as an argument,
so repeated calls on systems of the same structure do not compile again.

#### Scenario: A sweep of a one-dimensional map
- **WHEN** a user computes the spectrum of the logistic map for 10 values of `r` with 2000 iterations each, after one
  call
- **THEN** the 10 calls take less than 1 second together, and each exponent equals the exponent of a fresh call

### Requirement: Orbit diagrams and basins run as vectorised ensembles
`orbit_diagram` and `basin_stability` SHALL accept `vectorized=True` to run their sweep as one simulation of a
replicated system, with the same results as the sweep of separate runs.

#### Scenario: Vectorised period doubling
- **WHEN** a user builds the orbit diagram of the logistic map for 200 values of r in [2.8, 4] with `vectorized=True`
- **THEN** its points equal those of the serial sweep within 1e-12

#### Scenario: Vectorised basins
- **WHEN** a user estimates the basin stability of a bistable system over 200 initial conditions with
  `vectorized=True`
- **THEN** the fractions equal those of the serial runs with the same seed

### Requirement: Lyapunov spectra of laws with absolute values
`lyapunov_spectrum` SHALL differentiate `abs(x)` as `sign(x)`, so laws with absolute values have a spectrum.

#### Scenario: Sprott jerk system
- **WHEN** a user computes the spectrum of `SprottJerkSystem` until 200
- **THEN** it returns three finite exponents whose sum is negative

### Requirement: Scaling laws across sizes
`scaling(runs, kind='allometric')` SHALL fit the exponent of a quantity against system size across a set of runs,
read as a mapping from size to samples, and SHALL return the exponent with a bootstrap confidence interval.

#### Scenario: Known exponent
- **WHEN** a quantity generated as `N^1.15` times lognormal noise across 50 sizes is fitted
- **THEN** the 95% confidence interval contains 1.15

### Requirement: Fixed points come in a fixed order
`stability` SHALL return fixed points sorted by their values in the order of the state names, the same in every
process, and exact solutions SHALL convert to floats without solver noise.

#### Scenario: Logistic map
- **WHEN** the fixed points of the logistic map are found in two fresh processes
- **THEN** both return the same points in the same order, and the point at 0 is exactly 0.0

### Requirement: Aperiodic orbits have period 0
`orbit_diagram` SHALL report a period of 0 for a parameter value whose kept values never repeat within the rounding.

#### Scenario: Chaotic logistic map
- **WHEN** an orbit diagram of the logistic map keeps 64 values at r equal to 3.9
- **THEN** its period is 0, while r equal to 3.2 gives 2

### Requirement: Lattices coarse grain over blocks
`coarse_grain` SHALL coarse grain a lattice configuration over blocks of a given side, by majority with a tie rule or
by mean, for one snapshot or every frame of a trajectory.

#### Scenario: Majority over 3 by 3 blocks
- **WHEN** a 9 by 9 spin lattice coarse grains with `blocks=3` and `rule='majority'`
- **THEN** the result is 3 by 3 and every cell holds the sign of the sum of its block

#### Scenario: Mean over blocks
- **WHEN** a lattice coarse grains with `rule='mean'`
- **THEN** every cell holds the mean of its block

#### Scenario: Blocks that do not divide the lattice
- **WHEN** a 10 by 10 lattice coarse grains with `blocks=3`
- **THEN** an error names the side and the block size

### Requirement: Orbit diagrams scan many values quickly
`orbit_diagram` SHALL scan the logistic map over 1200 parameter values with 1000 transient steps in a few seconds when
it runs vectorized, and its docstring SHALL point large scans to the vectorized run.

#### Scenario: Scan time
- **WHEN** the vectorized scan runs on a laptop
- **THEN** it finishes in under 10 seconds with the same orbits as separate runs

### Requirement: Stable fits report the S0 location
`fit_law(law='stable')` SHALL report the location in the S0 parametrisation, and SHALL report a skewness of 0 when the
index is 2.

#### Scenario: Index near 1
- **WHEN** samples of stable laws of index 0.98 and 1.02 with S0 location 0 are fitted
- **THEN** both fitted locations are close to 0

#### Scenario: Gaussian data
- **WHEN** normal samples are fitted
- **THEN** the index is close to 2 and the skewness is 0

### Requirement: Branching ratio estimates are quiet
`events(kind='branching')` SHALL print nothing to standard output, and SHALL pass the messages of its estimator on as
Python warnings.

#### Scenario: Captured output
- **WHEN** a branching ratio is estimated while standard output is captured
- **THEN** nothing is captured

### Requirement: Box counting checks grids and counts points
`scaling(kind='box-counting')` SHALL raise an `AnalysisError` for a grid thinner than its largest box, and SHALL count
an array of points at every box size.

#### Scenario: Thin grid
- **WHEN** a grid of shape `(1, 512)` is box counted with boxes larger than 1
- **THEN** an `AnalysisError` names the side that is too short

#### Scenario: Points of a Sierpinski triangle
- **WHEN** 50000 chaos game points of the Sierpinski triangle are box counted as points
- **THEN** the dimension is close to log 3 / log 2

### Requirement: Detrended fluctuation windows are unique
`scaling(kind='detrended-fluctuation')` SHALL use default window sizes without duplicates.

#### Scenario: Default windows
- **WHEN** the fluctuation of a short series is estimated with the default windows
- **THEN** the reported window sizes are strictly increasing
