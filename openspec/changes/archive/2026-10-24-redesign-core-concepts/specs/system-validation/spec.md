# Spec Delta

## ADDED Requirements

### Requirement: Invalid spaces are reported as issues
Validation SHALL report every space whose own check fails, such as a lattice with a wrong shape, boundary, or
geometry, as an issue with a stable code, the path of the system that holds the space, the reason, and a fix, and SHALL
never skip a space whose check fails.

#### Scenario: Lattice with a bad boundary
- **WHEN** a system holds a lattice space whose boundary is not one of the supported boundaries
- **THEN** the report holds an error with the code `space-invalid` that names the boundary and lists the supported ones

#### Scenario: Invalid space among other errors
- **WHEN** a system has an invalid lattice and an unbound role
- **THEN** the report holds both errors, and compiling the system raises the validation error instead of an error from
  inside the compiler
