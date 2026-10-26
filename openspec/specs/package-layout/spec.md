# package-layout Specification

## Purpose
Separates the domain-agnostic framework, the core, from the domain model library, so every model is built only from
the general interface and the interface never depends on a model.

## Requirements

### Requirement: Dependencies run from formalisms to engine to core
Within `core` and within `engine`, each package SHALL import only the packages listed before it. `core` SHALL import
neither `engine` nor `formalisms`, and `engine` SHALL import `core` and not `formalisms`.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.engine.solvers` imports from `skcomplex.engine.backends`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.components` imports from `skcomplex.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: The core never depends on models
No module of `skcomplex.core` SHALL import `skcomplex.models`, and every core feature SHALL work without the model
library imported.

#### Scenario: Core without models
- **WHEN** a user builds, compiles, simulates, and analyses a system written from scratch, and `skcomplex.models` was
  never imported
- **THEN** every step works and `skcomplex.models` stays out of the loaded modules

#### Scenario: Planted reverse import
- **WHEN** a core module imports from `skcomplex.models`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Models use only the public core
Modules of `skcomplex.models` SHALL import only public names of the packages of `skcomplex.core`,
`skcomplex.engine`, and `skcomplex.formalisms`, and third-party libraries, never private modules or names.

#### Scenario: Private core import
- **WHEN** a model module imports a private module, such as `skcomplex.engine.compile._compile`
- **THEN** the import contracts of the checks session fail and name the module and the import

### Requirement: Model packages are self-contained
Every subfield package of `skcomplex.models` SHALL be self-contained: it SHALL NOT import any other model package, of
its own field or another, except the shared realizations of its own field in `skcomplex.models.<field>.realizations`.
A component that more than one field needs SHALL be domain-agnostic and live in the core.

#### Scenario: Import across model packages
- **WHEN** a module of `skcomplex.models.biology.collective_behaviour` imports from
  `skcomplex.models.physics.mechanics`, or from `skcomplex.models.biology.ecology`
- **THEN** the import contracts of the checks session fail and name the module and the import

#### Scenario: Realizations of the own field
- **WHEN** a module of `skcomplex.models.physics.mechanics` imports from `skcomplex.models.physics.realizations`
- **THEN** the import contracts of the checks session pass

#### Scenario: Removing a model package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other model package still imports, and its models still pass the conformance suite

### Requirement: The test tree mirrors the source tree
Every test module SHALL test one module of `skcomplex` and SHALL live at the path that mirrors it, so
`tests/core/<package>/test_<module>.py` tests `skcomplex.core.<package>._<module>`, and
`tests/models/<field>/<subfield>/test_<module>.py` tests the module of that model package. No test module SHALL check
repository conventions, documentation tooling, or documentation data.

#### Scenario: Test without a mirrored module
- **WHEN** the test modules under `tests` are listed
- **THEN** each one names a module of `skcomplex` at the mirrored path, and none lies outside `tests/core` and
  `tests/models`

### Requirement: The test session runs the examples of docstrings and documents
The test session SHALL run every docstring example of `skcomplex` through pytest's doctest collection, and every Python
block of `README.md` and the user guide through `pytest-markdown-docs`, with no test module for either. A block that
builds on earlier blocks of its document SHALL be marked as their continuation.

#### Scenario: Failing docstring example
- **WHEN** a docstring example of a core module prints a value other than the one it shows
- **THEN** the test session fails and names the module and the example

#### Scenario: Failing user guide block
- **WHEN** a Python block of the user guide raises an error
- **THEN** the test session fails and names the document and the block

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
