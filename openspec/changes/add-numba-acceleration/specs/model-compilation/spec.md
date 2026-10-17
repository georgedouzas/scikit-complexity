# Spec Delta

## ADDED Requirements

### Requirement: Laws compile to a chosen backend
Compiling a system SHALL accept a backend, `numba` by default and `numpy` as the reference, and both backends SHALL
produce the same numeric functions for the same system.

#### Scenario: Backends agree
- **WHEN** an ideal gas with pair interactions is compiled with both backends and evaluated on the same state
- **THEN** the drift, boundary, and update results agree to within 1e-12 relative error

#### Scenario: Unknown backend
- **WHEN** a user asks for a backend that is not registered
- **THEN** a named error is raised that lists the registered backends

### Requirement: The Numba backend is faster on large groups
The Numba backend SHALL evaluate the functions of a large group faster than the NumPy backend.

#### Scenario: Million-member gas
- **WHEN** one step of an ideal gas of 1000000 members is evaluated with both backends after compilation
- **THEN** the Numba backend takes at most a third of the time of the NumPy backend

### Requirement: Compiled callables run inside kernels
A law value that is a Numba-compiled function SHALL be called from inside the compiled kernels, and a plain Python
callable SHALL still work through a Python path.

#### Scenario: Compiled callable law
- **WHEN** a rate law value is a Numba-compiled function of time and member values
- **THEN** the system compiles with the Numba backend and its results equal those of the NumPy backend

#### Scenario: Python callable law
- **WHEN** a rate law value is a plain Python callable and the backend is Numba
- **THEN** the system compiles and runs, with that law evaluated in Python

### Requirement: Compiled kernels are cached
The Numba backend SHALL cache compiled kernels on disk, keyed by the lowered expressions, so compiling the same system
in a new process reuses them.

#### Scenario: Second compilation
- **WHEN** the same system is compiled in two separate processes
- **THEN** the second compilation is at least five times faster than the first
