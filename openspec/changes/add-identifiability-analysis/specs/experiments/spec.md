## ADDED Requirements

### Requirement: Identifiability is an experiment function
The experiments layer SHALL hold `identifiability(system, data, bounds, *, method, ...)`, with `method` one of
`'profile'` and `'refits'`, which returns a frozen `IdentifiabilityResult` with the best-fit estimates and a verdict
per parameter, and SHALL raise `ValidationError` naming both methods for any other value.

#### Scenario: Unknown method
- **WHEN** a user calls `identifiability` with `method='fisher'`
- **THEN** it raises `ValidationError`, whose message names `'profile'` and `'refits'`

### Requirement: Profile likelihood gives intervals and verdicts
With `method='profile'`, `identifiability` SHALL profile the negative Gaussian log-likelihood over each parameter
within its bounds, re-optimising the others, and SHALL return each profile, the likelihood-ratio interval at `level`,
and the verdict `'identifiable'` when both ends of the interval lie inside the bounds, else `'not identifiable'`.

#### Scenario: Logistic growth through saturation
- **WHEN** a user profiles `r` and `K` of the logistic growth system on 30 points that reach saturation with 2% noise
- **THEN** both verdicts are `'identifiable'`, and each 95% interval holds the true value

#### Scenario: Data before saturation
- **WHEN** a user profiles `r` and `K` on points that end while the population is below a tenth of `K`
- **THEN** the verdict of `K` is `'not identifiable'`, and its interval reaches the upper bound

### Requirement: Refits under noise give relative errors
With `method='refits'`, `identifiability` SHALL simulate the best fit at the data times, add Gaussian noise of each
relative level in `noise` to `n` copies, refit every copy, and return the average relative error of every parameter
per level, with the verdict `'strong'` at most the level, `'weak'` at most ten times it, and `'none'` above.

#### Scenario: Saturating logistic data
- **WHEN** a user runs refits of the logistic growth system with 100 copies at the noise level 0.01 on data that reach
  saturation
- **THEN** the verdicts of `r` and `K` are `'strong'` or `'weak'`

#### Scenario: Reproducible refits
- **WHEN** a user runs the same refits twice with seed 3
- **THEN** both results hold the same relative errors

### Requirement: Virtual populations keep plausible parameters
The experiments layer SHALL hold `virtual_population(system, bounds, outputs, constraints, *, n, ...)`, which draws `n`
parameter rows from the priors, runs each, computes the named outputs of each trajectory, and returns a frozen
`VirtualPopulationResult` with the plausible rows, whose outputs lie inside the constraints, and the acceptance rate.

#### Scenario: Plausible patients
- **WHEN** a user samples 500 patients of the immuno-oncology QSP system with the constraint that the baseline tumour
  diameter lies between 1 and 10 centimetres
- **THEN** every plausible row has a baseline diameter in that range, and the acceptance rate equals their share of 500

#### Scenario: No plausible patient
- **WHEN** no row satisfies the constraints
- **THEN** `virtual_population` raises `ExperimentError`, whose message names the constraint that rejected most rows

### Requirement: Virtual populations match observed values by inclusion
Given `target`, observed values of one output, `virtual_population` SHALL select among the plausible rows with the
probability of inclusion proportional to the density of the observed values over the density of the plausible
population at each row's output, and SHALL hold the selected rows and their probabilities in the result.

#### Scenario: Match an observed distribution
- **WHEN** a user samples 2000 patients of the QSP system with a target of 100 observed log ratios of CD8 to
  regulatory T cells drawn from a normal distribution
- **THEN** the two-sample Kolmogorov-Smirnov statistic between the selected outputs and the target is below 0.15

### Requirement: Individuals are calibrated one by one
The experiments layer SHALL hold `calibrate_individuals(system, data, bounds, *, known=None, ...)`, which calls
`calibrate` on the data of every individual, with that individual's known model parameters set first, and returns a
frozen `IndividualCalibrationResult` with the calibration of every individual and a table of their estimates.

#### Scenario: Digital twins of virtual patients
- **WHEN** a user calibrates the tumour growth rate and the T cell killing rate of five virtual patients of the QSP
  system, each with its own dose, from their simulated tumour diameters with 2% noise
- **THEN** every estimate lies within 10% of that patient's true value, and the table holds one row per patient

#### Scenario: Known parameter that is also fitted
- **WHEN** a user names a parameter both in `known` for an individual and in `bounds`
- **THEN** `calibrate_individuals` raises `ValidationError`, whose message names the parameter and the individual
