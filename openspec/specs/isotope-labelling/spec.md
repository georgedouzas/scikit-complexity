# isotope-labelling Specification

## Purpose

Lets a user of the system library attach atom transitions to the reactions of a metabolic network, simulate how labelled
atoms of a tracer spread through it at metabolic and isotopic steady state, and estimate the fluxes from measured
labelling patterns and fluxes, with confidence intervals and a goodness-of-fit test.

## Requirements

### Requirement: Reactions declare atom transitions
`AtomMapSpec` of `skcomplex.systems.biology.systems_biology` SHALL pair each reactant and product occurrence of a
reaction with a string of atom letters, so that every traced atom of a product names the reactant atom it comes from.
Invalid maps SHALL raise a named error that names the reaction and the offending metabolite or letter.

#### Scenario: Valid map
- **WHEN** a reaction consumes `AcCoA` with letters `ab` and `OAC` with letters `cdef` and produces `Cit` with letters
  `fedbac`
- **THEN** the map records that atom 1 of `Cit` comes from atom 4 of `OAC` and atom 5 of `Cit` from atom 1 of `AcCoA`

#### Scenario: Unbalanced letters
- **WHEN** a product names a letter that no reactant carries
- **THEN** checking raises a named error that names the reaction and the letter

#### Scenario: Inconsistent atom count
- **WHEN** one reaction gives `Glu` five letters and another gives it four
- **THEN** checking the analysis raises a named error that names the metabolite and both reactions

### Requirement: Labelling is simulated at isotopic steady state
`labelling(source, kind='simulation')` SHALL compute the mass isotopomer distributions of chosen fragments, given the
forward and backward fluxes and the tracer fractions, by the elementary metabolite unit decomposition with one sparse
linear system per unit size, and its distributions SHALL be non-negative and sum to 1.

#### Scenario: Published TCA example
- **WHEN** the TCA example of Antoniewicz et al. (2007) runs with `v1 = v2 = 100`, `v3 = v4 = v5 = v8 = v9 = 50`,
  fumarase forward 125 and backward 75, an acetyl-CoA tracer of 50% unlabelled, 25% [2-13C], and 25% [1,2-13C], and
  unlabelled aspartate
- **THEN** the distribution of glutamate atoms 1 to 5 is `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]` within
  5e-5, and matches mfapy within 1e-10

#### Scenario: Unlabelled tracer
- **WHEN** every tracer is unlabelled
- **THEN** every simulated distribution is `[1, 0, ..., 0]` within 1e-12

#### Scenario: Symmetric metabolite
- **WHEN** succinate is declared symmetric and labelled only at atom 1 by its producing reaction
- **THEN** the fumarate it feeds carries half of that label at atom 1 and half at atom 4

### Requirement: Fluxes are estimated from labelling data
`labelling(source, kind='estimation')` SHALL estimate the fluxes by weighted least squares over measured distributions
and measured fluxes, from one or more parallel tracer experiments, with the fluxes parametrised by free net and
exchange fluxes so every balance holds exactly, from several seeded feasible starts.

#### Scenario: Noise-free data
- **WHEN** the TCA example is fitted to the noise-free glutamate distributions of two parallel experiments, 50%
  [1-13C] and 50% [1,2-13C] acetyl-CoA, with `v1` fixed at 100
- **THEN** every net flux is recovered within 1e-4 relative error, and the residual is below 1e-10

#### Scenario: Reproducible starts
- **WHEN** the same estimation runs twice with the same seed
- **THEN** both runs set the same estimates and the same residual

#### Scenario: Data that cannot be fitted
- **WHEN** a measured flux contradicts the steady-state balances
- **THEN** the residual lies above the acceptance range of the goodness-of-fit test, and `accepted` is false

### Requirement: Estimates have confidence intervals and a goodness-of-fit test
An estimation SHALL set a 95% interval for every net and exchange flux, linearised by default and profiled on request,
and SHALL test the fit against the chi-square distribution whose degrees of freedom are the measurements minus the free
fluxes.

#### Scenario: Noisy data
- **WHEN** the TCA example is fitted to distributions with Gaussian noise of standard deviation 0.002 at seed 0
- **THEN** every net flux's profile interval contains its true value, and the test accepts the fit at the 5% level

#### Scenario: Unidentifiable exchange
- **WHEN** the fumarase exchange has no effect on the measured distributions within the fit tolerance
- **THEN** its profile interval reaches a bound, and `identifiable` marks it as not identifiable
