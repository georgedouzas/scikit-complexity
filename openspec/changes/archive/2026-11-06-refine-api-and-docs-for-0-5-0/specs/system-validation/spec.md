# Spec Delta

## MODIFIED Requirements

### Requirement: Compilation and simulation validate first
A compiler and `Simulation` SHALL validate in `prepare` before doing any other work, raise one `ValidationError` listing
every error of the report, and emit warnings as named warnings. A valid system SHALL compile without any further
compatibility error.

#### Scenario: Error before compilation
- **WHEN** an invalid system is passed to `Simulation`
- **THEN** a `ValidationError` is raised before any kernel is generated, and its message lists every error with its
  path and fix
