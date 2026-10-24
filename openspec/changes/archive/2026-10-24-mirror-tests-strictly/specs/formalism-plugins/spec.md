# Spec Delta

## MODIFIED Requirements

### Requirement: The shared core never imports a formalism
No module of `skcomplex.core` outside `skcomplex.core.formalisms` SHALL import a formalism, and a system built only from
the shared core SHALL compile and simulate with no formalism imported.

#### Scenario: Shared core alone
- **WHEN** a user builds, compiles, and simulates a logistic growth model written with `Element`, `Rate`, and `System`
- **THEN** the run succeeds and no module of `skcomplex.core.formalisms` is loaded

#### Scenario: Planted import
- **WHEN** a module of `skcomplex.core.compile` imports from `skcomplex.core.formalisms.lattices`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Formalisms declare their dependencies
A formalism SHALL declare the formalisms it builds on, may import only those and the shared core, and the declared
dependencies SHALL form no cycle.

#### Scenario: Declared dependency
- **WHEN** `fields` declares that it builds on `lattices`
- **THEN** importing `skcomplex.core.formalisms.fields` registers both formalisms

#### Scenario: Undeclared import
- **WHEN** a module of `skcomplex.core.formalisms.kinetics` imports from `skcomplex.core.formalisms.agents` and
  `kinetics` does not declare `agents`
- **THEN** the import contracts of the checks session fail and name both formalisms
