# Spec Delta

## MODIFIED Requirements

### Requirement: Conservation laws use finite volumes
Fields with declared fluxes SHALL discretise by finite volumes with limited reconstruction, conserving their totals and
resolving shocks without spurious oscillations. The finite-volume solver SHALL run them in PyClaw when its optional
extra is installed and supports the field, and in the in-house kernel otherwise, from the same flux laws.

#### Scenario: Burgers shock
- **WHEN** Burgers' equation develops a shock from a sine wave on a periodic domain
- **THEN** the total stays constant within 1e-12 relative error, and the solution has no new extrema near the shock

#### Scenario: Fallback without the extra
- **WHEN** the dam break runs with `FiniteVolumeSolver()` and PyClaw is not installed
- **THEN** it runs on the in-house kernel, and the solver reports the native path

#### Scenario: Pinned path
- **WHEN** a user runs the dam break with `FiniteVolumeSolver(path='native')` and PyClaw installed
- **THEN** the run uses the in-house kernel and reports the native path

## ADDED Requirements

### Requirement: The finite-volume solver declares its support
The finite-volume solver SHALL declare that it supports continuous time, lattice spaces, and flux, rate, and boundary
laws, so the compile report lists a system it cannot run before compilation.

#### Scenario: Finite volumes on a network
- **WHEN** a system whose field lives on a network space is checked with `FiniteVolumeSolver`
- **THEN** the report holds a `solver-unsupported` issue saying that the solver does not support the network space

### Requirement: Wave speeds use closed-form eigenvalues
The in-house finite-volume kernel SHALL compute wave speeds from eigenvalues of the flux Jacobians derived once in
closed form, and SHALL use numerical eigenvalues only for Jacobians without one.

#### Scenario: Same speeds
- **WHEN** the wave speeds of the dam break are computed from the closed forms and numerically on the same states
- **THEN** they agree within 1e-12 relative error

#### Scenario: Faster dam break
- **WHEN** the dam break on 1600 cells runs to time 0.5 with closed-form speeds and slicing, and then with numerical
  eigenvalues and `np.take`
- **THEN** the first run is faster by at least half of the speed-up the re-measurement of this change records and
  gives the same solution within 1e-10

#### Scenario: No closed form
- **WHEN** a field declares a flux whose Jacobian has no closed-form eigenvalues
- **THEN** the kernel falls back to numerical eigenvalues and reports it, and the run conserves the totals
