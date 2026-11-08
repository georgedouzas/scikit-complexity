## ADDED Requirements

### Requirement: Neural calibration methods
`calibrate` SHALL accept the methods `'npe'`, `'nle'`, and `'nre'` beside the other calibration methods. Each SHALL
train a neural estimator of the posterior, the likelihood, or the likelihood ratio on the summaries of `budget` runs
from prior draws, and SHALL return posterior samples of the observed data in a `CalibrationResult`.

#### Scenario: Neural posterior of a stochastic SIR
- **WHEN** a user calibrates `beta` and `gamma` of the stochastic SIR system with `method='npe'` and a budget of 2000
  runs on the infected counts of one run at `beta = 0.3` and `gamma = 0.1`
- **THEN** the central 90% interval of the samples of each parameter holds its true value

#### Scenario: Neural likelihood
- **WHEN** a user calibrates the logistic growth system with `method='nle'` on data with 5% noise
- **THEN** the posterior median of every parameter lies within 10% of its true value

#### Scenario: Unknown method lists the neural methods
- **WHEN** a user calls `calibrate` with `method='flow'`
- **THEN** it raises `ValidationError`, whose message lists `'npe'`, `'nle'`, and `'nre'` among the allowed methods

### Requirement: Neural estimators are reused across observations
A `CalibrationResult` of a neural method SHALL hold the trained estimator as `estimator`, and SHALL hold `None` there
for every other method. `calibrate` with `estimator=` a previous result and `budget=0` SHALL sample the posterior of
new data from that estimator without running the system.

#### Scenario: Amortized posterior
- **WHEN** a user calibrates the stochastic SIR system with `method='npe'`, then calls `calibrate` on other observed
  counts with `estimator=` that result and `budget=0`
- **THEN** the second call runs no simulation and returns samples whose median differs from the first

#### Scenario: Estimator of another system
- **WHEN** a user passes as `estimator` a result fitted on parameters other than the bounds of the call
- **THEN** `calibrate` raises `ValidationError`, whose message names the parameters of both

### Requirement: Neural posteriors refine over sequential rounds
`calibrate` with a neural method and `rounds` greater than one SHALL split `budget` over the rounds, draw the runs of
every round after the first from the current posterior of the observed data, and SHALL then hold an estimator tied to
that observation, which `estimator=` reuse SHALL reject for other data.

#### Scenario: Sequential rounds narrow the posterior
- **WHEN** a user calibrates the two belief parameters of the Brock-Hommes system with `method='npe'`, a budget of 2000,
  and once with one round and once with four rounds, from seed 0
- **THEN** the posterior standard deviation of every parameter after four rounds is at most that after one round

#### Scenario: A pretrained estimator is adapted
- **WHEN** a user passes `estimator=` an amortized result with `budget=200` and `rounds=2` on new data
- **THEN** the estimator is refined on 200 new runs and the result reports both the inherited and the new runs

### Requirement: Neural calibration learns summaries
`calibrate` with a neural method SHALL take `embedding`, `None` to use the summaries as given, or the name of a learned
embedding network, `'mlp'` or `'cnn'`, trained jointly with the estimator on the series at the data times.

#### Scenario: Learned embedding
- **WHEN** a user calibrates the stochastic SIR system with `method='npe'` and `embedding='cnn'` without `summary`
- **THEN** the result holds samples and `diagnostics` reports the size of the learned summary

### Requirement: Neural diagnostics test coverage
A `CalibrationResult` of a neural method SHALL hold in `diagnostics` the training and validation losses of the
estimator. `posterior_coverage` with a neural method SHALL train one amortized estimator for all truths and SHALL add
the expected coverage curve of the TARP test to the coverage per level.

#### Scenario: Amortized coverage
- **WHEN** a user calls `posterior_coverage` on the logistic growth system with `method='npe'` and 100 truths
- **THEN** the system runs only for the training budget and the 100 data sets, and the empirical coverage at the level
  0.9 lies within 0.15 of 0.9 for every parameter

#### Scenario: Training losses
- **WHEN** a user calibrates with `method='nre'`
- **THEN** `diagnostics` holds the training and validation loss of every epoch
