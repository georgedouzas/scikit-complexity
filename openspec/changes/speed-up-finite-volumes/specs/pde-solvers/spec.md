# Spec Delta

## ADDED Requirements

### Requirement: Wave speeds use closed-form eigenvalues
The in-house finite-volume kernel SHALL compute wave speeds from eigenvalues of the flux Jacobians derived once in
closed form, and SHALL use numerical eigenvalues only for Jacobians without one.

#### Scenario: Same speeds
- **WHEN** the wave speeds of the dam break are computed from the closed forms and numerically on the same states
- **THEN** they agree within 1e-12 relative error

#### Scenario: Faster dam break
- **WHEN** the dam break on 1600 cells runs to time 0.5 with closed-form speeds, and then with numerical eigenvalues,
  both with slicing
- **THEN** the first run is faster by at least half of the speed-up the re-measurement of this change records and
  gives the same solution within 1e-10

#### Scenario: Burgers shock unchanged
- **WHEN** Burgers' equation develops a shock from a sine wave on a periodic domain with the faster kernel
- **THEN** the total stays constant within 1e-12 relative error, and the solution has no new extrema near the shock

#### Scenario: No closed form
- **WHEN** a field declares a flux whose Jacobian has no closed-form eigenvalues
- **THEN** the kernel falls back to numerical eigenvalues and reports it through `closed_form`, and the run conserves
  the totals
