## ADDED Requirements

### Requirement: The package holds four subpackages
The package SHALL expose exactly four subpackages, which together hold every domain-agnostic package and the models:
- `skcomplex.core` holds what a system is and what it compiles to: `base`, `space`, `laws`, `components`, `system`,
  and `problems`.
- `skcomplex.engine` holds what works on a system: `compile`, `solvers`, `backends`, `runtime`, `viz`,
  `experiments`, and `analysis`.
- `skcomplex.formalisms` holds the formalisms.
- `skcomplex.models` holds the domain models.

#### Scenario: Framework imports
- **WHEN** a user imports `ElementComponent` from `skcomplex.core.components`, `Simulation` from
  `skcomplex.engine.runtime`, `RK4Solver` from `skcomplex.engine.solvers`, `Problem` from `skcomplex.core.problems`,
  and `NumericDomain` from `skcomplex.core.base`
- **THEN** every import succeeds

### Requirement: The core base holds the base classes of the framework
`skcomplex.core.base` SHALL define `BaseComponent`, `BaseInteraction`, `BaseSpace`, `BaseTime`, `BaseLaw`,
`BaseDomain`, `BaseProblem`, and `BaseSolver`, and the helpers they are declared with, in modules grouped by concept.
It SHALL hold no name that only one formalism uses, and SHALL import no other package of `skcomplex`.

#### Scenario: Base classes
- **WHEN** a user imports the eight base classes from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Formalism types
- **WHEN** a user imports `HilbertDomain` and `StructureDomain` from `skcomplex.core.base`
- **THEN** the import fails, and they import from `skcomplex.formalisms.quantum` and `skcomplex.formalisms.rewriting`

### Requirement: No base class is another base class
No base class SHALL derive from another. `BaseComponent`, `BaseInteraction`, `BaseSpace`, `BaseTime`,
`BaseSolver`, and `BaseBackend` SHALL derive from `sklearn.base.BaseEstimator` alone, and `BaseLaw`, `BaseDomain`,
`BaseProblem`, and `BaseSystemCompiler` from no class. `System` SHALL derive from `BaseEstimator` alone.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another

#### Scenario: System and interaction
- **WHEN** a test checks `System` and `BaseInteraction` against `BaseComponent`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: A class is named after what it is
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `System`, and `Schedule`,
SHALL end with that concept's name: `Component`, `Interaction`, `Space`, `Time`, `Law`, `Domain`, `Problem`,
`Solver`, `Backend`, `Observer`, `Visual`, `Clock`, `Design`, `Surrogate`, `Pass`, or `Schedule`. This SHALL hold
in the library, its models, its tests, and its documents, and for names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.laws`, `RK4Solver` from `skcomplex.engine.solvers`,
  `MeshSpace` from `skcomplex.formalisms.fields`, `ElementComponent` from `skcomplex.core.components`, and
  `ExternalSolver` from `skcomplex.engine.solvers`
- **THEN** every import succeeds

#### Scenario: Model names
- **WHEN** every class that the model library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from

### Requirement: The core problems package holds what a system compiles to
`skcomplex.core.problems` SHALL hold the compiler's `Problem`, the kernels, reductions, neighbourhoods, and printing
that a problem's numeric functions are written from, the `Lowering` a backend returns, and `BaseBackend` with the
registry of backends. It SHALL import no package of `skcomplex.engine`, and the engine SHALL register the loader that
imports its backends.

#### Scenario: Backend base
- **WHEN** a user imports `BaseBackend`, `Lowering`, and `Problem` from `skcomplex.core.problems`
- **THEN** every import succeeds, and `NumbaBackend` of `skcomplex.engine.backends` derives from `BaseBackend`

#### Scenario: Backends load on demand
- **WHEN** a system is compiled with the backend `'numba'` and `skcomplex.engine.backends` was never imported
- **THEN** the registry of backends imports it through the loader the engine registered, and the compile succeeds

### Requirement: The runtime reads the base problem
The runtime SHALL run any problem through what `BaseProblem` holds, and SHALL read the compiler's fields only of a
`Problem`. The quantum and rewriting problems SHALL derive from `BaseProblem` alone, with the fields they use.

#### Scenario: Problem of a formalism
- **WHEN** a quantum system and a rewriting system are simulated and recorded
- **THEN** both runs succeed, and neither problem is a subclass of `skcomplex.core.problems.Problem`

### Requirement: A system compiler is not a pass
`BaseSystemCompiler` of `skcomplex.engine.compile` SHALL be the base of the compilers that claim a whole system of a
formalism and compile it to a problem, and SHALL be unrelated to `BasePass`, whose subclasses are steps of one
compilation.

#### Scenario: Quantum compiler
- **WHEN** a test checks `QuantumCompiler` against `BaseSystemCompiler` and `BasePass`
- **THEN** it derives from the first and not from the second

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseComponent`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every time from `BaseTime`, every law from `BaseLaw`, every domain from `BaseDomain`, every problem from
`BaseProblem`, every solver from `BaseSolver`, and every backend from `BaseBackend`. Validation SHALL read a
domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks `ElementComponent` and `GroupComponent` against `BaseComponent`,
  `PairPotentialInteraction` against `BaseInteraction`, `QuantumProblem` against `BaseProblem`, `RK4Solver` and
  `LammpsSolver` against `BaseSolver`, and `NumbaBackend` against `BaseBackend`
- **THEN** every check holds

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain, and `skcomplex.engine.compile` imports nothing from
  `skcomplex.formalisms.quantum`

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and initialized with a `BaseProblem` of its form
- **THEN** it advances the problem without importing `skcomplex.engine`

## REMOVED Requirements

### Requirement: The core holds every domain-agnostic package
**Reason**: It checked that old framework paths fail to import, and the package keeps no reference to its former
layout.
**Migration**: The requirement "The package holds four subpackages" states the layout.
