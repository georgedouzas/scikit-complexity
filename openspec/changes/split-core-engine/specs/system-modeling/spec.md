# Spec Delta

## MODIFIED Requirements

### Requirement: Core packages mirror the system definition
The modelling packages of `skcomplex.core` SHALL be `system`, `components`, `interactions`, `space`, which holds space
and time, and `laws`, each exporting the general primitives of its part of a system. A primitive that belongs to one
formalism SHALL live in that formalism's package under `skcomplex.formalisms`, never in a modelling package.
`skcomplex.core.model` SHALL not exist.

#### Scenario: Primitives by part
- **WHEN** a user imports `System`, `Element`, `Interaction`, `LatticeSpace`, `DiscreteTime`, and `Rate`
- **THEN** they come from `core.system`, `core.components`, `core.interactions`, `core.space`, `core.space`, and
  `core.laws`

#### Scenario: Formalism primitives by formalism
- **WHEN** a user imports `Particle`, `PairPotential`, `Rule`, and `Field`
- **THEN** they come from `formalisms.particles`, `formalisms.particles`, `formalisms.agents`, and
  `formalisms.fields`

### Requirement: Backends live by their reach
Every general backend SHALL live in `skcomplex.engine.backends`, a backend that serves one formalism SHALL live in that
formalism, and `compile` and `solvers` SHALL hold no backend.

#### Scenario: Backends in one package
- **WHEN** the test suite lists the registered backends
- **THEN** NumPy and Numba are defined in `skcomplex.engine.backends`, Mesa in `skcomplex.formalisms.agents`, and
  LAMMPS in `skcomplex.formalisms.particles`
