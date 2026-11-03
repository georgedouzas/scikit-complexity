# Spec Delta

## Purpose

Keep the runtime dependencies of the package actively released, so a stale library cannot hold back a Python version or a
platform, and keep results stable when a stale library is replaced.

## ADDED Requirements

### Requirement: Runtime dependencies are actively released

Every runtime dependency SHALL have a release on the package index within the last 24 months, or the project SHALL record
why that dependency may stay and when the exception is next reviewed.

#### Scenario: A dependency passes the limit

- **WHEN** a runtime dependency has had no release for more than 24 months and has no recorded exception
- **THEN** the dependency check before a release names it, and the project replaces it or records an exception

#### Scenario: Adding a dependency

- **WHEN** a contributor proposes a new runtime dependency
- **THEN** the proposal states its last release date, and a library past the limit is not added without a recorded exception

### Requirement: Replacing a dependency keeps its results

Replacing a runtime dependency SHALL keep the names and meaning of the public results that rely on it, and their values
within numerical tolerance of reference values recorded from the previous library.

#### Scenario: Mean squared displacement

- **WHEN** a user computes the mean squared displacement of the reference positions
- **THEN** the displacement over every lag, the exponent, and the diffusion coefficient agree with the recorded reference
  values within 1e-9 relative tolerance

#### Scenario: Kramers-Moyal coefficients

- **WHEN** a user computes the Kramers-Moyal coefficients of the reference series with the same bins and bandwidth
- **THEN** the positions, the drift, and the diffusion agree with the recorded reference values within 1e-6 relative
  tolerance

#### Scenario: Stable matching

- **WHEN** a stable matching runs on the reference preferences and capacities
- **THEN** it gives the same pairs as the requester-optimal matching the previous library gave, and no requester and
  provider prefer each other over their partners

### Requirement: A replaced dependency is gone

A replaced dependency SHALL NOT remain in the source, the manifest, the lock file, or the project documentation.

#### Scenario: After a replacement

- **WHEN** a contributor searches the repository for a replaced dependency outside the archived changes
- **THEN** nothing is found
