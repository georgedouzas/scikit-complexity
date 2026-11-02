# Spec Delta

## ADDED Requirements

### Requirement: Series inputs of any length compile
An interpolated series input SHALL compile and run on every backend for any number of data points, with the same
values and breakpoints as for a short series.

#### Scenario: Long held series
- **WHEN** a law reads `series(s.t, times, values, 'previous')` with 1000 points at times 0 to 999
- **THEN** the system compiles on the NumPy and the Numba backends without a syntax or recursion error, and the input
  at time `k + 0.5` equals the value of point `k` for every `k`

#### Scenario: Long linear series
- **WHEN** a law reads a linear series of 1000 points of `sin` at times 0 to 999
- **THEN** the input at every time equals NumPy's linear interpolation of the data within 1e-12, and every data time
  is a breakpoint of the solver
