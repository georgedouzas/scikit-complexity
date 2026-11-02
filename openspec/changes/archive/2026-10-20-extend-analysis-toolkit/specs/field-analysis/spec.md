# Spec Delta

## Purpose

Measures the spatial structure of lattices and fields of any dimension, from spins, cellular automata, Monte Carlo
samples, coarse-grained agents, or PDE solutions: structure factors, correlation lengths, coarsening, patterns,
clusters, percolation, and finite-size scaling.

## ADDED Requirements

### Requirement: Structure factors and correlation lengths
The analysis layer SHALL compute the radially averaged structure factor of a field on a periodic grid, and its
correlation length by the second-moment estimate or by a fit of `C(r) ~ r^-a e^(-r / xi)`, averaged over frames.

#### Scenario: Ising lattice above its critical temperature
- **WHEN** a user computes the correlation length of a 128 × 128 Ising lattice in equilibrium at temperature 2.5 and 3
- **THEN** each lies within 15 percent of the exact `1 / (-ln tanh K - 2 K)` with `K = 1 / T`

#### Scenario: Same correlation length on a reaction-diffusion field
- **WHEN** a user computes the correlation length of the stationary noisy linear reaction-diffusion field
  `u_t = D lap u - k u + noise` with `D = 4` and `k = 1`
- **THEN** the result lies within 10 percent of `sqrt(D / k) = 2`

### Requirement: Coarsening is measured by the domain size
The analysis layer SHALL compute the characteristic domain size of every frame from the first moment of the structure
factor or from the first zero of the correlation function, and fit its growth exponent in time.

#### Scenario: Two coarsening laws
- **WHEN** a user measures domain growth after a quench of a 256 × 256 Glauber Ising lattice to temperature 1, and of
  a 128 × 128 Cahn-Hilliard field from a symmetric mixture
- **THEN** the Ising exponent lies within 0.1 of 1/2, as preasymptotic corrections lower it at this size, and the
  Cahn-Hilliard exponent within 0.05 of 1/3

### Requirement: Pattern wavelengths are measured
The analysis layer SHALL compute the dominant wavelength and the orientation order of a pattern from the peak of its
structure factor, and the giant number fluctuations of a density field over boxes of growing size.

#### Scenario: Turing patterns on two domains
- **WHEN** a user measures the wavelength of a Swift-Hohenberg stripe pattern with `q0 = 1` and of a Brusselator
  pattern with `b = 1.95`, in its Turing regime below the Hopf onset
- **THEN** the first lies within 5 percent of `2 pi` and the second within 10 percent of the critical wavelength from
  the linear stability of its homogeneous state

#### Scenario: Giant number fluctuations
- **WHEN** a user measures number fluctuations of the coarse-grained density of an ideal gas and of Vicsek agents in
  their ordered phase
- **THEN** the gas exponent of the standard deviation against the mean lies within 0.05 of 1/2, and the Vicsek
  exponent exceeds 0.6

### Requirement: Clusters and percolation observables
The analysis layer SHALL compute cluster size distributions of grids and of graph components, the mean cluster size
without the largest, the strength of the largest cluster, and the probability of a spanning cluster over an ensemble.

#### Scenario: Cluster sizes at the threshold on two kinds of data
- **WHEN** a user fits the cluster size distribution of site percolation on periodic 1024 × 1024 grids at 0.5927 above
  size 100, and of the components of Erdős-Rényi graphs of 10^5 nodes at mean degree 1
- **THEN** the grid exponent lies within 0.1 of 187/91 and the graph exponent within 0.1 of 5/2

#### Scenario: Forest fire clusters
- **WHEN** a user computes the strength of the largest cluster of burnt trees of a forest fire model over tree densities
  from 0.4 to 0.8
- **THEN** it rises from below 0.05 to above 0.5, and the sharpest rise lies within 0.03 of 0.5927

### Requirement: Finite-size scaling collapses curves
The analysis layer SHALL compute Binder cumulants over an ensemble, locate their crossing between sizes, and collapse
curves of several sizes on scaled variables with given or fitted exponents and a quality measure of the collapse.

#### Scenario: Ising critical point
- **WHEN** a user computes Binder cumulants of Ising lattices of sides 16, 32, and 64 across temperatures
- **THEN** the crossings lie within 1 percent of `2 / ln(1 + sqrt 2)`

#### Scenario: Same collapse on two transitions
- **WHEN** a user collapses the Ising susceptibility with `nu = 1` and `gamma / nu = 7/4` over sides 16, 32, and 64,
  and the spanning probability of site percolation with `nu = 4/3` over the same sides
- **THEN** both collapses have a quality better than with exponents 20 percent away, and fitting `nu` recovers each
  within 10 percent
