## Purpose
Gives every analysis one contract: an estimator that stores its settings, checks them, prepares against the data or the
system it measures, runs, and keeps its results as fitted attributes, so a modeller learns one way to measure anything.

## ADDED Requirements

### Requirement: Every analysis is an estimator with three verbs
Every analysis SHALL be a class whose constructor only stores its settings, whose `check()` validates them, whose
`prepare(source)` reads the data or the system it measures, and whose `run()` computes and returns the analysis itself.
No analysis SHALL be a public function.

#### Scenario: Transfer entropy of a trajectory
- **WHEN** a user runs `TransferEntropyAnalysis(lag=1, bins=8).check().prepare(trajectory).run()` on a trajectory of two
  coupled logistic maps
- **THEN** the call returns the analysis, and its `transfer_entropy_` holds one value per ordered pair of series

#### Scenario: Settings are checked, not constructed
- **WHEN** a user builds an entropy analysis with a negative embedding dimension
- **THEN** construction succeeds, and `check()` raises a named error that names the setting and the value

#### Scenario: Run before prepare
- **WHEN** a user calls `run()` on an analysis that was checked but never prepared
- **THEN** a named error says that the analysis must be prepared with a source first

### Requirement: Results are fitted attributes
An analysis SHALL keep every result in a public attribute whose name ends in `_`, set by `prepare` or `run`, and SHALL
NOT return a separate result object. Its settings SHALL stay readable and settable through `get_params` and
`set_params`.

#### Scenario: Power-law fit
- **WHEN** a user runs a power-law fit analysis on avalanche sizes drawn from a power law of exponent 1.5
- **THEN** `exponent_` lies within 0.1 of 1.5, and `minimum_`, `statistic_`, and `p_value_` are set

#### Scenario: Cloned analysis
- **WHEN** a user clones a prepared analysis with scikit-learn's `clone`
- **THEN** the clone has the same settings and none of the fitted attributes

### Requirement: An analysis reads trajectories, arrays, graphs, and systems
`prepare(source)` SHALL accept a trajectory, a NumPy array, a NetworkX graph, or a system, as the analysis needs, and
SHALL extract the data it measures from a trajectory itself. Settings SHALL select the variables to read by name.

#### Scenario: Same analysis on a trajectory and an array
- **WHEN** a user prepares a structure factor analysis once on an Ising trajectory and once on the array of its last
  frame
- **THEN** both runs give the same `structure_factor_`

#### Scenario: Named variables
- **WHEN** a user prepares a series analysis with `variables=['prey.n']` on a Lotka-Volterra trajectory
- **THEN** the analysis reads only the series of `prey.n`

#### Scenario: Source of the wrong kind
- **WHEN** a user prepares an agent analysis on an ODE trajectory
- **THEN** a named error says that the trajectory has no agent group and lists the data kinds it has

### Requirement: Analyses of a model take the system
An analysis that measures a model rather than recorded data, such as stability, linearization, or Lyapunov exponents,
SHALL take the system as its source and SHALL compile it itself.

#### Scenario: Fixed point stability
- **WHEN** a user prepares a stability analysis on a Lotka-Volterra system and runs it
- **THEN** `fixed_points_` holds the coexistence point, and `eigenvalues_` holds its purely imaginary pair

### Requirement: Families share one class with a choice argument
Analyses that compute the same kind of quantity and differ only by a formula SHALL be one class, selected by a string
setting that `check()` validates against the choices the class lists.

#### Scenario: Entropy family
- **WHEN** a user runs `EntropyAnalysis(kind=kind)` on the same series for each of `'shannon'`, `'renyi'`,
  `'tsallis'`, `'permutation'`, `'sample'`, `'approximate'`, and `'spectral'`
- **THEN** every run sets `entropy_`, and the permutation entropy of white noise is within 0.02 of 1

#### Scenario: Unknown choice
- **WHEN** a user checks `EntropyAnalysis(kind='unknown')`
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
- **THEN** the analysis checks, prepares, and runs through the base's verbs

### Requirement: Analyses are general operations across domains
Every analysis class SHALL compute one mathematical operation, named after it, and SHALL read every data kind on which
the operation is defined, so the same class with the same settings applies to the data of any field. A choice SHALL be
a variant of the operation, never a quantity of one field. The analysis SHALL check in `prepare` that the source holds
a data kind it reads, and SHALL raise a named error that lists the kinds it reads otherwise.

#### Scenario: Alignment on three domains
- **WHEN** a user runs the alignment analysis of kind `mean-vector` on the headings of a Vicsek flock, on the spins of an
  XY lattice, and of kind `mean-phase` on the phases of Kuramoto oscillators
- **THEN** each run sets the order of every frame between 0 and 1, with the same class and the same settings names

#### Scenario: Moment ratio of any sample
- **WHEN** a user runs the distribution analysis of kind `moment-ratio` on magnetisation samples of an Ising model and
  on returns of a market model
- **THEN** both runs set the fourth-to-squared-second moment ratio of every sample

#### Scenario: Source without a readable kind
- **WHEN** a user prepares the alignment analysis on a trajectory with no agents, no angle field, and no phase series
- **THEN** a named error lists the data kinds the analysis reads and the data kinds the trajectory holds

