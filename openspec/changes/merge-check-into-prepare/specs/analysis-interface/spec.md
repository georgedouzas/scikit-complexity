## RENAMED Requirements

- FROM: `### Requirement: Every analysis is an estimator with three verbs`
- TO: `### Requirement: Every analysis is an estimator with two verbs`

## MODIFIED Requirements

### Requirement: Every analysis is an estimator with two verbs
Every analysis SHALL be a class whose constructor only stores its settings, whose `prepare(source)` validates them and
reads the data or the system it measures, and whose `run()` computes and returns the analysis itself.
No analysis SHALL be a public function.

#### Scenario: Transfer entropy of a trajectory
- **WHEN** a user runs `TransferEntropyAnalysis(lag=1, bins=8).prepare(trajectory).run()` on a trajectory of two
  coupled logistic maps
- **THEN** the call returns the analysis, and its `transfer_entropy_` holds one value per ordered pair of series

#### Scenario: Settings are checked, not constructed
- **WHEN** a user builds an entropy analysis with a negative embedding dimension
- **THEN** construction succeeds, and `prepare` raises a named error that names the setting and the value

#### Scenario: Run before prepare
- **WHEN** a user calls `run()` on an analysis that was never prepared
- **THEN** a named error says that the analysis must be prepared with a source first

### Requirement: Families share one class with a choice argument
Analyses that compute the same kind of quantity and differ only by a formula SHALL be one class, selected by a string
setting that `prepare` validates against the choices the class lists.

#### Scenario: Entropy family
- **WHEN** a user runs `EntropyAnalysis(kind=kind)` on the same series for each of `'shannon'`, `'renyi'`,
  `'tsallis'`, `'permutation'`, `'sample'`, `'approximate'`, and `'spectral'`
- **THEN** every run sets `entropy_`, and the permutation entropy of white noise is within 0.02 of 1

#### Scenario: Unknown choice
- **WHEN** a user prepares `EntropyAnalysis(kind='unknown')`
- **THEN** a named error lists the kinds the class takes

### Requirement: The analysis surface is classes only
The public surface of the analysis package SHALL hold only analysis classes and the base a user subclasses. Every name
SHALL end with `Analysis`, except the base `BaseAnalysis`.

#### Scenario: Surface listing
- **WHEN** a user lists the public names of the analysis package
- **THEN** every name is a class whose name ends with `Analysis`, and no name is a function

#### Scenario: A user analysis
- **WHEN** a user subclasses `BaseAnalysis`, implements the hooks to read a series and compute its mean, and runs it on
  a trajectory
- **THEN** the analysis prepares and runs through the base's verbs
