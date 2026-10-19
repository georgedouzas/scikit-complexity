# Spec Delta

## Purpose

Measures stochastic dynamics from jump paths, noisy series, transition matrices, and ensembles: first passage,
stationary laws, counting statistics, irreversibility, large deviations, Markov models, aging, response, and extremes.

## ADDED Requirements

### Requirement: First-passage and return times are measured
The analysis layer SHALL return the first-passage times of paths to a threshold or a set of states, the return times
to a level, the durations of laminar phases, and the hitting probabilities of absorbing states, over one path or an
ensemble.

#### Scenario: Drifted diffusion and gambler's ruin
- **WHEN** a user measures the first passage of 2000 Brownian paths with drift 0.5 to a barrier at 5, and of 2000 fair
  gambler's ruin walks from 3 to 0 or 10
- **THEN** the mean passage time lies within 5 percent of 10, and the mean duration within 5 percent of `3 (10 - 3)`

#### Scenario: Return times on two domains
- **WHEN** a user fits the tail of the first-return times of a symmetric random walk and of the laminar phases of an
  on-off intermittent map at its threshold
- **THEN** both fitted exponents lie within 0.1 of 3/2

#### Scenario: Fixation of a mutant
- **WHEN** a user measures the fixation probability of one neutral mutant in 5000 Moran runs of size 20
- **THEN** the result lies within 0.01 of 1/20

### Requirement: Stationary laws and relaxation follow from transitions
The analysis layer SHALL compute the stationary distribution, the spectral gap, and the relaxation times of a
transition matrix or a rate matrix, SHALL estimate the matrix from a discrete path, and SHALL test detailed balance.

#### Scenario: Ehrenfest urn and random walk on a ring
- **WHEN** a user analyses the rate matrix of an Ehrenfest urn of 20 balls and the lazy random walk on a cycle graph of
  11 nodes
- **THEN** the urn's stationary law is binomial with p = 1/2 within 1e-10, and the walk's gap equals
  `(1 - cos(2 pi / 11)) / 2` within 1e-10

#### Scenario: Detailed balance
- **WHEN** a user tests detailed balance on long paths of a reversible birth-death chain and of a driven three-state
  cycle
- **THEN** the birth-death chain passes and the driven cycle fails with a net probability current different from 0

### Requirement: Counting statistics and moments of fluctuations
The analysis layer SHALL compute Fano factors of event counts in windows or of copy numbers, relative fluctuations
against system size, and Kramers-Moyal drift and diffusion coefficients estimated from the increments of a series.

#### Scenario: Poisson and bursty production
- **WHEN** a user computes Fano factors of the counts of a zeroth-order production run with Gillespie's algorithm and
  of the mRNA copy numbers of bursty gene expression with mean burst size 4
- **THEN** the first lies within 0.05 of 1 and the second within 0.3 of 5

#### Scenario: Ornstein-Uhlenbeck coefficients
- **WHEN** a user estimates the Kramers-Moyal coefficients of 10^6 steps of `dx = -x dt + sqrt(2) dW`
- **THEN** the drift lies within 0.05 of `-x` and the diffusion within 0.05 of 1 for `|x| < 2`

#### Scenario: Relative fluctuations
- **WHEN** a user measures the standard deviation over mean of the particle number in subvolumes of an ideal lattice
  gas
- **THEN** it scales with the subvolume as `V^-1/2` within 0.05 in the exponent

### Requirement: Entropy production measures irreversibility
The analysis layer SHALL compute the entropy production rate and the stochastic entropy of every path from the laws of
a jump or Langevin system, the rate of a discrete path from its transition counts, and SHALL check the fluctuation
theorem on the stochastic entropy samples.

#### Scenario: Same entropy production on two domains
- **WHEN** a user computes the entropy production of a Brownian particle on a ring driven by force 1 with friction 1
  at temperature 1, and of a jump process on a ring of 5 states with rates 2 forward and 1 backward
- **THEN** the particle gives 1 within 5 percent and the jump process `(2 - 1) ln 2` within 5 percent, per unit time

#### Scenario: Estimate from counts
- **WHEN** a user estimates the entropy production of the driven ring from the transition counts of its path alone
- **THEN** the result lies within 5 percent of `ln 2`, and a reversible chain gives below 0.01

#### Scenario: Fluctuation theorem
- **WHEN** a user checks the fluctuation theorem on 10,000 runs of the driven particle over time 1
- **THEN** the slope of `ln P(s) / P(-s)` against `s` lies within 0.1 of 1

### Requirement: Large deviations of time-additive observables
The analysis layer SHALL compute the scaled cumulant generating function of a counting or current observable exactly
from a tilted rate matrix or by blocks of a long series, the rate function by its Legendre-Fenchel transform or from
histograms of an order parameter at several sizes, and histograms reweighted by an exponential tilt.

#### Scenario: Poisson counting
- **WHEN** a user computes the generating function of the count of a Poisson process with rate 2 from its tilted
  generator and from blocks of one long run
- **THEN** both equal `2 (e^k - 1)` for `|k| <= 1` within 2 percent, and the rate function at 3 equals
  `3 ln(3 / 2) - 1` within 2 percent

#### Scenario: Gallavotti-Cohen symmetry
- **WHEN** a user computes the generating function of the current of the driven ring with rates 2 and 1
- **THEN** it satisfies `lambda(k) = lambda(-k - ln 2)` within 1e-8

#### Scenario: Mean-field order parameter
- **WHEN** a user estimates the rate function of the magnetization of the mean-field Ising model above its critical
  temperature from histograms at sizes 50, 100, and 200
- **THEN** it agrees with the exact rate function within 0.02 for `|m| < 0.5`

### Requirement: Markov state models are estimated
The analysis layer SHALL estimate Markov state models from discrete paths at a lag time, with implied timescales,
metastable sets, committors between two sets, and mean first-passage times between sets.

#### Scenario: Known three-state chain
- **WHEN** a user estimates a model at lag 1 from 10^5 steps of a known three-state chain
- **THEN** every transition probability lies within 0.02 of the true one, and the implied timescales are flat in the
  lag within 5 percent

#### Scenario: Double well on two domains
- **WHEN** a user estimates models of an overdamped particle in a double well, discretized in 50 bins, and of a
  bistable Schlögl reaction network on its copy numbers
- **THEN** both find two metastable sets, and the slowest implied timescale is within 20 percent of the mean
  first-passage time between the wells

### Requirement: Two-time correlations, aging, and response
The analysis layer SHALL compute two-time correlations and responses from an ensemble, test their dependence on the
waiting time alone, collapse distributions on the ratio of times, and compute fluctuation-dissipation ratios.

#### Scenario: Equilibrium has no aging
- **WHEN** a user computes the two-time autocorrelation of an equilibrium Ornstein-Uhlenbeck ensemble at waiting times
  10 and 100
- **THEN** both curves are `exp(-tau)` within 0.03 and the fluctuation-dissipation ratio lies within 0.1 of 1

#### Scenario: Aging on two domains
- **WHEN** a user computes the two-time correlations of a Glauber Ising quench below its critical temperature and of
  Bouchaud's trap model with `T / T_g = 0.5`
- **THEN** in each the correlation at fixed `tau` grows with the waiting time, and the curves collapse onto
  `tau / t_w` within 0.05

#### Scenario: Trap model distribution
- **WHEN** a user collapses the trapping-time distributions `P(tau, t)` of the trap model at ages 10^3, 10^4, and 10^5
- **THEN** `t P(tau, t)` against `tau / t` collapses within 10 percent

### Requirement: Extremes, records, and stable limits
The analysis layer SHALL fit generalized extreme-value laws to block maxima, count records and their waiting times, fit
stable laws to sums, and compare samples with a named law by a goodness-of-fit test.

#### Scenario: Extreme-value classes
- **WHEN** a user fits block maxima of exponential, Pareto with exponent 2, and uniform samples
- **THEN** the shape parameter is 0 within 0.05, `1/2` within 0.1, and `-1` within 0.1

#### Scenario: Records of independent values
- **WHEN** a user counts the records in 10,000 sequences of 1000 independent values
- **THEN** the mean count lies within 1 percent of the harmonic number of 1000

#### Scenario: Boltzmann law on two domains
- **WHEN** a user compares the kinetic energies of a two-dimensional ideal gas and the money of a random-exchange
  economy, both after relaxation, with an exponential law
- **THEN** neither rejects the law at the 0.05 level, and the fitted temperatures equal the initial energy per molecule
  and the money per agent within 2 percent

#### Scenario: Reweighted magnetization
- **WHEN** a user reweights the magnetization histogram of the mean-field Ising model of size 100 at field 0 to field
  0.05
- **THEN** the reweighted mean lies within 0.02 of the mean of a run at field 0.05
