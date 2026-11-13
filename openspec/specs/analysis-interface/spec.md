# analysis-interface Specification

## Purpose
Gives every analysis one contract: a public function of a trajectory, an array, or a graph that takes keyword settings
and returns its values as a frozen result, so a modeller learns one way to measure anything.

## Requirements

### Requirement: An analysis reads trajectories, arrays, graphs, and systems
An analysis function SHALL accept as its source a trajectory, a NumPy array, a NetworkX graph, or a system, as it needs, and
SHALL extract the data it measures from a trajectory itself. Settings SHALL select the variables to read by name.

#### Scenario: Same analysis on a trajectory and an array
- **WHEN** a user runs a structure factor analysis once on an Ising trajectory and once on the array of its last
  frame
- **THEN** both runs give the same `structure_factor`

#### Scenario: Named variables
- **WHEN** a user runs a series analysis with `variables=['prey.n']` on a Lotka-Volterra trajectory
- **THEN** the analysis reads only the series of `prey.n`

#### Scenario: Source of the wrong kind
- **WHEN** a user runs an agent analysis on an ODE trajectory
- **THEN** a named error says that the trajectory has no agent group and lists the data kinds it has

### Requirement: Analyses of a system take the system
An analysis that measures a system rather than recorded data, such as stability, linearization, or Lyapunov exponents,
SHALL take the system as its source and SHALL compile it itself.

#### Scenario: Fixed point stability
- **WHEN** a user runs `stability` on a Lotka-Volterra system
- **THEN** `fixed_points` of its result holds the coexistence point, and `eigenvalues` holds its purely imaginary pair

### Requirement: Families share one class with a choice argument
Analyses that compute the same kind of quantity and differ only by a formula SHALL be one function, selected by a string
setting that the function validates against the choices it lists.

#### Scenario: Entropy family
- **WHEN** a user calls `entropy(series, kind=kind)` on the same series for each of `'shannon'`, `'renyi'`,
  `'tsallis'`, `'permutation'`, `'sample'`, `'approximate'`, and `'spectral'`
- **THEN** every result sets `entropy`, and the permutation entropy of white noise is within 0.02 of 1

#### Scenario: Unknown choice
- **WHEN** a user calls `entropy(series, kind='unknown')`
- **THEN** a named error lists the kinds the function takes

### Requirement: Analyses are general operations across domains
Every analysis function SHALL compute one mathematical operation, named after it, and SHALL read every data kind on which
the operation is defined, so the same function with the same settings applies to the data of any field. A choice SHALL be
a variant of the operation, never a quantity of one field. The analysis SHALL check that the source holds
a data kind it reads, and SHALL raise a named error that lists the kinds it reads otherwise.

#### Scenario: Alignment on three domains
- **WHEN** a user runs the alignment analysis of kind `mean-vector` on the headings of a Vicsek flock, on the spins of an
  XY lattice, and of kind `mean-phase` on the phases of Kuramoto oscillators
- **THEN** each run sets the order of every frame between 0 and 1, with the same function and the same settings names

#### Scenario: Moment ratio of any sample
- **WHEN** a user runs the distribution analysis of kind `moment-ratio` on magnetisation samples of an Ising system and
  on returns of a market system
- **THEN** both runs set the fourth-to-squared-second moment ratio of every sample

#### Scenario: Source without a readable kind
- **WHEN** a user runs the alignment analysis on a trajectory with no agents, no angle field, and no phase series
- **THEN** a named error lists the data kinds the analysis reads and the data kinds the trajectory holds

### Requirement: Analyses are functions of results
Every analysis of a trajectory, an array, or a graph SHALL be a public function of that result with keyword settings,
in the role of a function of `sklearn.metrics`, and SHALL return its values as a frozen result. Settings worth reusing
SHALL be bound by one helper in the role of `make_scorer`, which returns a callable a function such as `simulate` or an
experiment accepts.

#### Scenario: Measuring a trajectory
- **WHEN** a user calls `lyapunov(trajectory, kind='largest', variables=['x'])`
- **THEN** it returns the exponents, with no object to prepare or run

#### Scenario: Reusing settings
- **WHEN** a user binds an analysis with fixed settings and passes it to a parameter sweep
- **THEN** the sweep applies it to the trajectory of every run

### Requirement: An analysis of a trajectory reports times in its units

An analysis that reports times or lags SHALL read the step between samples from the times of a trajectory it is
given. A step given as an argument SHALL apply only to arrays, which carry no times.

#### Scenario: Correlation of a trajectory with a step of a tenth

- **WHEN** `correlation` is run on a trajectory sampled every 0.1 time units
- **THEN** its lags are 0, 0.1, 0.2, and so on, and its correlation time is in time units

#### Scenario: Correlation of an array

- **WHEN** `correlation` is run on an array with `dt=0.5`
- **THEN** its lags are 0, 0.5, 1.0, and so on

### Requirement: The core analyses measure what several fields share
Every analysis of `skcomplex.core.analysis` SHALL measure a quantity that systems of several fields share. An analysis
of one field SHALL live in the common package of that field and SHALL follow the same contract as the core analyses:
a public function of a trajectory, a system, or data, with keyword settings and a frozen result. The discovery of
applicable analyses SHALL list the core analyses only.

#### Scenario: Discovery lists the core analyses
- **WHEN** a user calls `applicable_analyses` on a trajectory of a chain of quantum spins
- **THEN** the result lists neither `operator_measures` nor any metabolism analysis, and every analysis it lists is
  a public name of `skcomplex.core.analysis`

#### Scenario: Field analysis with the core contract
- **WHEN** a user calls `operator_measures(trajectory, kind='purity')` from `skcomplex.systems.physics.common.quantum`
  on a trajectory of a dephasing qubit
- **THEN** it returns a frozen result whose purity starts at 1 and never increases from frame to frame
