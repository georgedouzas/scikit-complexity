## ADDED Requirements

### Requirement: The core base holds the base classes of the core
`skcomplex.core.base` SHALL define `BaseComponent`, `BaseLaw`, `BaseSpace`, and `BaseDomain`, and the helpers they
are declared with: the errors, `Registry`, the kinds, `state`, `parameter`, `role`, `symbol`, `vector`, and the
numeric, categorical, and tensor domains. It SHALL hold no name that only the engine or a single formalism uses, and
SHALL import no other package of `skcomplex`.

#### Scenario: Base classes
- **WHEN** a user imports `BaseComponent`, `BaseLaw`, `BaseSpace`, and `BaseDomain` from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old base names are gone
- **WHEN** a user imports `Component` from `skcomplex.core.components`, `Law` from `skcomplex.core.laws`, or `Space`
  from `skcomplex.core.space`
- **THEN** the import fails

#### Scenario: Engine vocabulary
- **WHEN** a user imports `Form` and `Supports` from `skcomplex.core.base`, and then from `skcomplex.engine.compile`
- **THEN** the first import fails and the second succeeds

#### Scenario: Formalism types
- **WHEN** a user imports `HilbertSpace` and `Structure` from `skcomplex.core.base`
- **THEN** the import fails, and they import from `skcomplex.formalisms.quantum` and `skcomplex.formalisms.rewriting`

### Requirement: Every class of the core derives from a core base
Every component SHALL derive from `BaseComponent`, every law from `BaseLaw`, every space from `BaseSpace`, and every
domain from `BaseDomain`. Validation SHALL read a domain's kind from the domain, never from a list of domain
classes.

#### Scenario: Derived classes
- **WHEN** a test checks `Element`, `Group`, `Interaction`, and `System` against `BaseComponent`, `Rate` against
  `BaseLaw`, `LatticeSpace` against `BaseSpace`, and `Domain.REAL` against `BaseDomain`
- **THEN** every check holds

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertSpace` domain and the system is validated
- **THEN** validation reads its kind from the domain, and `skcomplex.engine.compile` imports nothing from
  `skcomplex.formalisms.quantum`

### Requirement: Solvers and backends derive from engine bases
Every solver SHALL derive from `BaseSolver` of `skcomplex.engine.solvers`. Every backend SHALL derive from
`BaseBackend` of `skcomplex.engine.compile`. No alias SHALL keep the names `Stepper` and `Backend`.

#### Scenario: Solver and backend bases
- **WHEN** a test checks `RK4` against `BaseSolver` and the numba backend against `BaseBackend`
- **THEN** both checks hold

#### Scenario: Old engine base names are gone
- **WHEN** a user imports `Stepper` from `skcomplex.engine.solvers` or `Backend` from `skcomplex.engine.compile`
- **THEN** the import fails
