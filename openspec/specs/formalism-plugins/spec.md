# formalism-plugins Specification

## Purpose
Lets each domain-agnostic formalism, such as lattice dynamics, rewriting, or quantum dynamics, add its laws, components,
compilation, solvers, backends, visuals, and checks to the core as one self-contained package, so the shared core never
depends on any formalism and a new formalism needs no change to existing core modules.

## Requirements

### Requirement: Formalisms live in one package each
Every formalism SHALL be one subpackage of `skcomplex.core.formalisms` that holds its laws, components, compilation,
solvers, backends, visuals, and validation checks. The formalisms SHALL be `agents`, `lattices`, `groups`, `rewriting`,
`quantum`, `fields`, `kinetics`, `fractional`, `topology`, `particles`, and `control`. A name
specific to one formalism SHALL be imported from that formalism's package.

#### Scenario: Formalism imports
- **WHEN** a user imports `LocalRule` from `skcomplex.core.formalisms.lattices`, `Production` from
  `skcomplex.core.formalisms.rewriting`, and `Generator` from `skcomplex.core.formalisms.quantum`
- **THEN** every import succeeds

#### Scenario: Old paths are gone
- **WHEN** a user imports `LocalRule` from `skcomplex.core.laws`
- **THEN** the import fails

### Requirement: The shared core never imports a formalism
No module of `skcomplex.core` outside `skcomplex.core.formalisms` SHALL import a formalism, and a system built only from
the shared core SHALL compile and simulate with no formalism imported.

#### Scenario: Shared core alone
- **WHEN** a user builds, compiles, and simulates a logistic growth model written with `Element`, `Rate`, and `System`
- **THEN** the run succeeds and no module of `skcomplex.core.formalisms` is loaded

#### Scenario: Planted import
- **WHEN** a module of `skcomplex.core.compile` imports from `skcomplex.core.formalisms.lattices`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: A formalism registers through one declaration
Each formalism SHALL describe everything it contributes in one declaration that its package registers when it is
imported: its law classes, component classes, compilation passes with their order, forms, solvers, backends, visuals,
and validation checks. The core SHALL resolve laws, forms, solvers, backends, and visuals only through these
registrations.

#### Scenario: Registered contributions are listed
- **WHEN** a user imports `skcomplex.core.formalisms.quantum` and lists the registered formalisms
- **THEN** the listing shows `quantum` with its law kinds, its form, and its solvers

#### Scenario: Unregistered law
- **WHEN** a system holds an instance of a law class that no formalism registered
- **THEN** validation reports an issue that names the law class and says that a formalism must register it

### Requirement: Adding a law kind changes no existing module
A new law class SHALL become usable by declaring its kind and traits and registering it in a formalism, with no edit to
any module of the shared core.

#### Scenario: Law from an outside formalism
- **WHEN** a test package defines a formalism with a new law kind, a compilation pass for it, and a solver, and
  registers it
- **THEN** a system using that law validates, compiles, and simulates without changes to `skcomplex.core`

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
