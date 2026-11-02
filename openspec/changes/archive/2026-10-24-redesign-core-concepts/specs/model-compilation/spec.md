# Spec Delta

## MODIFIED Requirements

### Requirement: The form is inferred from the law kinds
Compiling a validated system SHALL derive its form from the laws it lowers, and the form SHALL be an internal property
of the compiled problem rather than part of the system's definition. A system whose nested systems have different law
classes SHALL be split into partitions. Unsupported mixes SHALL be reported by validation before compilation, not by the
compiler, and validation and compilation SHALL decide partitions by the same rule, so a system that validates without an
error SHALL compile, and a system the compiler would reject SHALL fail validation.

#### Scenario: Deterministic ecology
- **WHEN** a system contains only rate laws
- **THEN** the compiled problem has the ODE form

#### Scenario: Stochastic epidemic
- **WHEN** a system contains only jump laws
- **THEN** the compiled problem has the jump process form

#### Scenario: Unsupported mix
- **WHEN** one system without children contains both update laws and rate laws
- **THEN** validation reports the mix, names the components by law class, and suggests moving them into nested systems,
  before the compiler runs

#### Scenario: Mix across subsystems
- **WHEN** one nested system contains only update laws and a sibling contains only rate laws
- **THEN** compiling produces a partitioned problem with a discrete map partition and an ODE partition

#### Scenario: Validation and compilation agree
- **WHEN** every registered model is validated and compiled
- **THEN** each model that validates without an error compiles, and no model fails compilation with a partition error

### Requirement: Laws compile to a chosen backend
Compiling a system SHALL accept a backend, `numba`, `numpy` as the reference, or `auto` by default, which picks `numba`
for systems whose groups hold at least 1000 members in total and `numpy` otherwise, and both backends SHALL produce the
same numeric functions for the same system. When the chosen backend cannot serve the form of a problem, or a compiled
kernel falls back to Python, compilation SHALL report it with a warning that names the backend used, the part
affected, and the reason, and the compiled problem SHALL record which backend runs each part.

#### Scenario: Backends agree
- **WHEN** an ideal gas with pair interactions is compiled with both backends and evaluated on the same state
- **THEN** the drift, boundary, and update results agree to within 1e-12 relative error

#### Scenario: Automatic choice
- **WHEN** a system with a group of 1000 members and a system with two single elements are compiled with `auto`
- **THEN** the first uses the Numba backend and the second uses the NumPy backend

#### Scenario: Unknown backend
- **WHEN** a user asks for a backend that is not registered
- **THEN** a named error is raised that lists the registered backends

#### Scenario: Backend substituted for a form
- **WHEN** a lattice model is compiled with `backend='numba'` and its form runs on NumPy kernels
- **THEN** compilation warns that the lattice form runs on NumPy, and the problem records NumPy for that form

#### Scenario: Kernel falls back to Python
- **WHEN** a law holds a reduction that the Numba printer cannot translate
- **THEN** compilation warns once, naming the law kind and the reduction, and the problem records which law kinds run
  compiled
