## MODIFIED Requirements

### Requirement: A simulation holds a default end time
`simulate(system)` SHALL run to the tuned `until` of the run settings of the system. An `until` passed to `simulate`
SHALL take precedence, and a system with neither SHALL raise a `ValidationError` that asks for one.

#### Scenario: Stored end time
- **WHEN** a user calls `simulate(system)` for a system class whose tuned `until` is 20
- **THEN** the run ends at time 20

#### Scenario: Passed end time wins
- **WHEN** a user calls `simulate(system, until=5)` for a system class whose tuned `until` is 20
- **THEN** the run ends at time 5

#### Scenario: No end time
- **WHEN** a user calls `simulate(system)` for a system with no tuned `until`
- **THEN** it raises a `ValidationError` that shows how to pass an end time

### Requirement: A system package plugs in its own compiler
A compiler defined outside the engine, on the public `BaseCompiler`, SHALL compile a system whose `compiler` run setting
names it, or a system passed to `simulate` with `compiler=`, and the run SHALL use the solver that compiler names,
without the engine importing the package. The compilers of the library's own forms SHALL stay private, and `simulate`
SHALL choose them from the form the system holds.

#### Scenario: Rewriting from the systems library
- **WHEN** `simulate(KochCurveSystem(), until=3)` runs without a compiler
- **THEN** the compiler of the rewriting form is chosen automatically, the run rewrites the word once per frame, and
  `skcomplex.core.engine` imports nothing from `skcomplex.systems`

#### Scenario: Quantum system without its compiler
- **WHEN** a quantum system runs through `simulate` without a compiler
- **THEN** the compiler of the operator form is chosen automatically and the run succeeds
