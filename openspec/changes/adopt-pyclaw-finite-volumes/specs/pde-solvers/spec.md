# Spec Delta

## MODIFIED Requirements

### Requirement: Conservation laws use finite volumes
Fields with declared fluxes SHALL discretise by finite volumes with limited reconstruction, conserving their totals and
resolving shocks without spurious oscillations. The default finite-volume solver SHALL run them in PyClaw when its
optional extra is installed and supports the field, and in the in-house scheme otherwise, from the same flux laws.

#### Scenario: Burgers shock
- **WHEN** Burgers' equation develops a shock from a sine wave on a periodic domain
- **THEN** the total stays constant within 1e-12 relative error, and the solution has no new extrema near the shock

#### Scenario: Fallback without the extra
- **WHEN** the dam break runs with the default finite-volume solver and PyClaw is not installed
- **THEN** it runs on the in-house scheme, and the solver reports the native backend

#### Scenario: Pinned path
- **WHEN** a user asks the finite-volume solver for the native backend with PyClaw installed
- **THEN** the run uses the in-house scheme and reports the native backend

## ADDED Requirements

### Requirement: Wave speeds use closed-form eigenvalues
The in-house finite-volume scheme SHALL compute wave speeds from eigenvalues of the flux Jacobians derived once in
closed form, and SHALL use numerical eigenvalues only for Jacobians without one.

#### Scenario: Same speeds
- **WHEN** the wave speeds of the dam break are computed from the closed forms and numerically on the same states
- **THEN** they agree within 1e-12 relative error

#### Scenario: Faster dam break
- **WHEN** the dam break on 1600 cells runs to time 0.5 with closed-form speeds and then with numerical eigenvalues
- **THEN** the closed-form run is at least 3 times faster and gives the same solution within 1e-10

#### Scenario: No closed form
- **WHEN** a field declares a flux whose Jacobian has no closed-form eigenvalues
- **THEN** the scheme falls back to numerical eigenvalues and reports it, and the run conserves the totals
