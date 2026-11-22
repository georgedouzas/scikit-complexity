## ADDED Requirements

### Requirement: Array updates of well-mixed groups
An `UpdateLaw(state, f, arrays=True)` of a group in a system without a space SHALL give its function the value of every
member as one array with one row per member and the components of a vector value on the last axis, and SHALL replace
it with the array the function returns, on the NumPy and Numba backends under discrete time.

#### Scenario: Swap of paired rows
- **WHEN** a group of 6 members with a vector state of 4 components and no space is updated by a function that swaps
  rows 0 and 1
- **THEN** the next frame holds the swapped rows and the other rows unchanged

#### Scenario: Wrong shape
- **WHEN** the function returns an array of another shape
- **THEN** the step raises `SimulationError`, whose message names the law and both shapes
