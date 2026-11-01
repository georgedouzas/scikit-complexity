# Spec Delta

## ADDED Requirements

### Requirement: Mode decomposition with control inputs
`ModeDecompositionAnalysis` SHALL take `inputs`, the names of the variables that are control inputs, and SHALL then fit
dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, on the variables or on the observables of its
dictionary. It SHALL set `operator_` and `input_matrix_`, and its `system_` SHALL take every input as a bounded adaptive
parameter that its rate laws read.

#### Scenario: Linear system with input
- **WHEN** the analysis with one input is prepared on runs of a known linear system driven by a random
  piecewise-constant input
- **THEN** `operator_` and `input_matrix_` equal the matrices of the system's one-sample map within 1e-6

#### Scenario: Input as a control
- **WHEN** the `system_` of the analysis is simulated with a schedule of its input parameter
- **THEN** the trajectory records the input at every frame, and the states follow the fitted map
