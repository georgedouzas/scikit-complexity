## ADDED Requirements

### Requirement: Analyses are functions of results
Every analysis of a trajectory, an array, or a graph SHALL be a public function of that result with keyword settings,
in the role of a function of `sklearn.metrics`, and SHALL return its values as a frozen result. Settings worth reusing
SHALL be bound by one helper in the role of `make_scorer`, which returns a callable a function such as `simulate` or an
experiment accepts.

#### Scenario: Measuring a trajectory
- **WHEN** a user calls `lyapunov_exponents(trajectory, states=['x', 'y', 'z'])`
- **THEN** it returns the exponents, with no object to prepare or run

#### Scenario: Reusing settings
- **WHEN** a user binds an analysis with fixed settings and passes it to a parameter sweep
- **THEN** the sweep applies it to the trajectory of every run

## REMOVED Requirements

### Requirement: Every analysis is an estimator with two verbs
**Reason**: Analyses measure results, so they become functions of the results, as metrics are in scikit-learn.
**Migration**: Call the function of the analysis on the trajectory or the array, with the settings that the
constructor took.

### Requirement: The analysis surface is classes only
**Reason**: The analysis package becomes a package of functions, with one binding helper and the result values.
**Migration**: Import the function with the analysis's name in snake case.
