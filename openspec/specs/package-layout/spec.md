# package-layout Specification

## Purpose
Separates the domain-agnostic framework, the core, from the domain model library, so every model is built only from
the general interface and the interface never depends on a model.

## Requirements

### Requirement: Dependencies run from formalisms to engine to core
The top-level packages SHALL import downward in the order `experiments`, `analysis`, `engine`, `core`, and the root
modules `base`, `types`, `specs`, and `errors` below them, so `analysis` never imports `experiments`, `engine` imports
neither, `core` imports none of them, and the root modules import no package. Within `core` and within `engine`, each
package SHALL import only the packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.engine.backends` imports from `skcomplex.engine.solvers`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.elements` imports from `skcomplex.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Analysis reaching experiments
- **WHEN** a module of `skcomplex.analysis` imports from `skcomplex.experiments`
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
The package SHALL expose exactly five subpackages, which together hold every domain-agnostic package and the models,
beside the root modules that every subpackage shares:
- `skcomplex.base`, `skcomplex.types`, `skcomplex.specs`, and `skcomplex.errors` hold the root class `BaseConfig`,
  the type aliases, the specs, and the errors.
- `skcomplex.core` holds what a system is: `base`, `spaces`, `laws`, `elements`, `interactions`, and `system`.
- `skcomplex.engine` holds what turns a system into a trajectory: `problems`, `compile`, `backends`, `solvers`,
  `runtime`, and `viz`.
- `skcomplex.analysis` holds what measures a trajectory, an array, or a system.
- `skcomplex.experiments` holds what runs many systems.
- `skcomplex.models` holds the domain models.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.elements`, `Simulation` from `skcomplex.engine.runtime`,
  `RK4Solver` from `skcomplex.engine.solvers`, `EntropyAnalysis` from `skcomplex.analysis`, and `Experiment` from
  `skcomplex.experiments`
- **THEN** every import succeeds

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
Every element and group SHALL derive from `BaseElement`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every law from `BaseLaw`, every problem from `BaseProblem`, every solver from `BaseSolver`, every backend
from `BaseBackend`, every analysis from `BaseAnalysis`, and every spec from `BaseSpec`, and every one of these bases
SHALL derive from `BaseConfig`. Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks every class whose name starts with `Base` in `skcomplex.core`, `skcomplex.engine`,
  `skcomplex.analysis`, `skcomplex.experiments`, and `skcomplex.specs` against `BaseConfig`
- **THEN** every check holds

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and prepared with a `BaseProblem` of its form
- **THEN** it advances the problem through `run`

### Requirement: The core base holds the bases of a system's parts
`skcomplex.core.base` SHALL hold seven modules: `_parts` with `BasePart`, `_elements` with `BaseElement`,
`_interactions` with `BaseInteraction`, `_laws` with `BaseLaw`, and the shared `_types`, `_errors`, and `_utils`. It SHALL define no other
base, and SHALL import no other package of `skcomplex`. The base of any other concept SHALL live in that concept's
package.

#### Scenario: Three bases
- **WHEN** a user imports `BasePart`, `BaseElement`, `BaseInteraction`, and `BaseLaw` from `skcomplex.core.base`
- **THEN** every import succeeds, and the modules of `skcomplex.core.base` are exactly `_parts`, `_elements`,
  `_interactions`, `_laws`, `_types`, `_errors`, and `_utils`

#### Scenario: Other bases
- **WHEN** a user imports `BaseSpace` and `BaseTime` from `skcomplex.core.space`, and `Layout` and `BaseProblem` from
  `skcomplex.core.problems`
- **THEN** every import succeeds, and none of them imports from `skcomplex.core.base`

### Requirement: Bases are unrelated
No base class SHALL derive from another, except that `BaseElement`, `BaseInteraction`, `BaseLaw`, and `System`
SHALL derive from `BasePart`, which derives from scikit-learn's `BaseEstimator`. `BaseSpace`, `BaseTime`, `BaseSolver`,
and `BaseBackend` SHALL derive from the framework's `BaseEstimator`, and `BaseDomain`, `BaseProblem`, and `BaseSystemCompiler` from no base.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another, except the bases of parts, which are subclasses of `BasePart`

#### Scenario: System and interaction
- **WHEN** a test checks `System` and `BaseInteraction` against `BaseElement`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: Classes are named after their base
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `System`, and `Schedule`, SHALL
end with that concept's name, such as `Element`, `Interaction`, `Space`, `Time`, `Law`, `Domain`, `Problem`,
`Solver`, or `Backend`, in the library, its models, its tests, its documents, and the names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.laws`, `Element` and `GroupElement` from
  `skcomplex.core.components`, and `RK4Solver` from `skcomplex.engine.solvers`
- **THEN** every import succeeds

#### Scenario: Model names
- **WHEN** every class that the model library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from, and none ends with `Component`

### Requirement: What every package shares lives in root modules
A module used by every subpackage SHALL live at the root of the package, not in one subpackage: `skcomplex.base`
SHALL define `BaseConfig`, the root of every configurable class; `skcomplex.errors` SHALL define every error and
warning; `skcomplex.specs` SHALL define the specs; and `skcomplex.types` SHALL define every type alias built from
Python, NumPy, SciPy, and Sage types, one alias per type, such as `Array` for every float array. An alias that names
a class of the library SHALL stay in the module of that class, and the models library SHALL keep its own aliases.

#### Scenario: Root imports
- **WHEN** a user imports `BaseConfig` from `skcomplex.base`, `ValidationError` from `skcomplex.errors`,
  `StateSpec` from `skcomplex.specs`, and `Array` from `skcomplex.types`
- **THEN** every import succeeds and none of these modules imports a subpackage

#### Scenario: One alias per type
- **WHEN** a module of the core, the engine, analysis, or experiments needs a float array
- **THEN** it imports `Array` from `skcomplex.types` instead of defining its own alias

### Requirement: Library packages hold general operations
The core, the engine, analysis, and experiments SHALL hold general operations named mathematically: every public class
SHALL name what it computes or simulates, and its choices SHALL name variants of that operation. A name taken from one
application field SHALL appear only in the models library, as a realization that subclasses the general class in that
field's package and fixes its settings.

#### Scenario: Field name in a library package
- **WHEN** a reviewer lists the public classes and the choices of the core, the engine, analysis, and experiments
- **THEN** none is named after a quantity of one application field, such as the Binder cumulant, the polar order, or
  the segregation index

#### Scenario: Field measure in the models library
- **WHEN** a physics model reports the Binder cumulant of its magnetisation
- **THEN** it uses a realization of the physics field that subclasses the general distribution analysis and only fixes
  its settings

### Requirement: The engine surface is its pipeline
Every package of `skcomplex.engine` SHALL re-export only classes: the bases a developer subclasses, the classes of the
pipeline from a system to a trajectory, and the frozen values a client reads, with `compile_system` the only public
function. Every helper of one class SHALL be a private method of that class.

#### Scenario: Engine surface listing
- **WHEN** a user lists the public names of every package of `skcomplex.engine`
- **THEN** every name is a class derived from a base of its package, or `compile_system`, and none is `validate` or a
  helper of a single pass, such as `values_of`, `member_of`, or `hilbert_of`

#### Scenario: A user solver
- **WHEN** a user subclasses the solver base, implements its hooks, and passes an instance to a simulation
- **THEN** the simulation runs with that solver

### Requirement: Every engine class follows the common API
Every class of the engine SHALL derive from exactly one base of its package, SHALL store only its settings in
`__init__`, SHALL validate them in `prepare`, SHALL take the one object it serves in `prepare(target)` and set the
attributes that end in `_`, and SHALL take the one input it acts on in `run(inputs)`. A base SHALL take no argument in
its verbs that its peers do not take, and a variation between subclasses SHALL be a setting.

#### Scenario: Compiling a system
- **WHEN** a user builds the system compiler, checks it, prepares it on a system, and runs it
- **THEN** `prepare` sets `report_` with the issues of the system, and `run` returns the compiled problem or raises the
  report's errors

#### Scenario: A user kernel
- **WHEN** a developer subclasses the kernel base for a new law kind, with the pass that builds it and a solver that
  runs it
- **THEN** the kernel is prepared on the problem and run on the state like every built-in kernel

### Requirement: Engine dataclasses are frozen values
Every dataclass of the engine SHALL be a frozen value with the hook `_check` and plain readers only, and a class that acts SHALL
be a class of the engine's tree instead.

#### Scenario: No acting dataclass
- **WHEN** a reviewer lists the dataclasses of the engine
- **THEN** every one is frozen and has no public verb besides `prepare` and `run`, and no method besides the hook `_check` and plain readers

### Requirement: The code imports what it annotates
No module SHALL import inside an `if TYPE_CHECKING:` block, and the import contracts SHALL check type-checking imports
like any other. A spec SHALL live in the lowest package that can import every type its fields name.

#### Scenario: No hidden imports
- **WHEN** a test searches the source for `TYPE_CHECKING`
- **THEN** it finds none, and lint-imports passes with type-checking imports checked

### Requirement: Findings are fixed, not suppressed
The source, the tests, and the documentation SHALL hold no `noqa`, `type: ignore`, or `nosec` comment, and the checks
session SHALL pass: black, ruff, lint-imports, mypy, bandit, interrogate, pydoclint, and deptry.

#### Scenario: No suppression comments
- **WHEN** a test searches the source, the tests, and the examples for `noqa`, `type: ignore`, and `nosec`
- **THEN** it finds none

### Requirement: The engine names no realization of a model package
The engine SHALL NOT list a compiler, a form, or a law kind other than those of its own packages. A form, its laws,
its compiler, and its solver that serve the models of one field SHALL live in that field of the models library.

#### Scenario: No realization in the engine
- **WHEN** a test searches the public names of `skcomplex.core`, `skcomplex.engine`, `skcomplex.specs`, and `skcomplex.types`
  for `KochElement`, `CantorElement`, `SierpinskiElement`, and `grigorchuk_group`
- **THEN** it finds none

#### Scenario: Rewriting in the core and the engine
- **WHEN** a user imports `WordElement` and `PatchElement` from `skcomplex.core.elements`, `ProductionLaw` and
  `SubstitutionLaw` from `skcomplex.core.laws`, and `RewritingCompiler` from `skcomplex.engine.compile`
- **THEN** every import succeeds, and none of these modules imports `skcomplex.models`
