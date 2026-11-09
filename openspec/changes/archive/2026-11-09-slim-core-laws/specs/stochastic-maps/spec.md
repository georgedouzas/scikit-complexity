## MODIFIED Requirements

### Requirement: Emission laws give hidden Markov sources
A transition law with a source SHALL set an observed state to the value drawn from the row of its matrix that the
source state selects. It SHALL read the source state's value after the same step, so the hidden and observed sequences
are aligned frame by frame. Its matrix SHALL be symbolic, and a function matrix with a source SHALL raise a
`ValidationError`.

#### Scenario: Aligned emission
- **WHEN** a hidden chain with two states emits through a transition law with the hidden state as its source and the
  identity matrix
- **THEN** the observed sequence equals the hidden sequence at every frame

#### Scenario: Emission statistics
- **WHEN** a hidden Markov source with emission matrix `[[0.7, 0.3], [0.2, 0.8]]` runs for 100000 steps
- **THEN** the frequency of each observed symbol given each hidden state matches its emission row within 0.01

#### Scenario: Function matrix with a source
- **WHEN** a transition law with a source is given a function as its matrix
- **THEN** preparing it raises a `ValidationError` that asks for a symbolic matrix

### Requirement: Markov matrices are checked
Compiling a transition law SHALL check that its matrix is square without a source, has one column per value of the
target alphabet and one row per value of the selecting state, has nonnegative entries, and has rows that sum to one
within 1e-9, using the parameter values at compile time. Changing a parameter of a running simulation SHALL apply the
same check to every matrix that the parameter enters.

#### Scenario: Row that does not sum to one
- **WHEN** a transition matrix has a row that sums to 0.9
- **THEN** compiling raises a named error that names the law, the row, and the sum

#### Scenario: Live change breaks a row
- **WHEN** a running simulation sets a parameter so that a matrix entry becomes negative
- **THEN** the change is rejected with a named error that names the parameter, the law, and the entry, and the
  simulation keeps the previous value
