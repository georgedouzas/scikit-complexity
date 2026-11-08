# package-layout Specification

## Purpose
Separates the domain-agnostic framework, the core, from the domain system library, so every system is built only from
the general interface and the interface never depends on a system.

## Requirements

### Requirement: Dependencies run from formalisms to engine to core
The packages of `skcomplex.core` SHALL import downward in the order `experiments`, `analysis`, `engine`, `components`,
and the modules `base`, `types`, `specs`, and `errors` of `skcomplex.core` below them, so `analysis` never imports
`experiments`, `engine` imports neither, `components` imports none of them, and those modules import no package.
Within `components` and within `engine`, each package SHALL import only the packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.core.engine.backends` imports from `skcomplex.core.engine.solvers`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.components.elements` imports from `skcomplex.core.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Analysis reaching experiments
- **WHEN** a module of `skcomplex.core.analysis` imports from `skcomplex.core.experiments`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: The core never depends on systems
No module of `skcomplex.core` SHALL import `skcomplex.systems`, and every feature of the core SHALL work without the
catalog imported.

#### Scenario: Core without systems
- **WHEN** a user builds, compiles, simulates, and analyses a system written from scratch, and `skcomplex.systems` was
  never imported
- **THEN** every step works and `skcomplex.systems` stays out of the loaded modules

#### Scenario: Planted reverse import
- **WHEN** a module of the core imports from `skcomplex.systems`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Systems use only the public core
Modules of `skcomplex.systems` SHALL import only public names of the packages of `skcomplex.core` and third-party
libraries, never private modules or names.

#### Scenario: Private core import
- **WHEN** a module of `skcomplex.systems` imports a private module, such as `skcomplex.core.engine.compile._compile`
- **THEN** the import contracts of the checks session fail and name the module and the import

### Requirement: System packages are self-contained
Every subfield package of `skcomplex.systems` SHALL be self-contained: it SHALL NOT import any other subfield package,
of its own field or another, except the `common` package of its own field, `skcomplex.systems.<field>.common`, which
holds the components that several systems of that field share. A component that more than one field needs SHALL be
copied into each field, or, when it is domain-agnostic, live in the core.

#### Scenario: Import across system packages
- **WHEN** a module of `skcomplex.systems.biology.collective_behaviour` imports from
  `skcomplex.systems.physics.mechanics`, or from `skcomplex.systems.biology.ecology`
- **THEN** the import contracts of the checks session fail and name the module and the import

#### Scenario: Realizations of the own field
- **WHEN** a module of `skcomplex.systems.physics.mechanics` imports from `skcomplex.systems.physics.common`
- **THEN** the import contracts of the checks session pass

#### Scenario: Removing a system package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other subfield package still imports, and its systems still pass the conformance suite

### Requirement: The test tree mirrors the source tree
Every test module SHALL test one module of `skcomplex` and SHALL live at the path that mirrors it, so
`tests/core/<package>/test_<module>.py` tests `skcomplex.core.<package>._<module>`, and
`tests/systems/<field>/<subfield>/test_<module>.py` tests the module of that subfield package. No test module SHALL
check repository conventions, documentation tooling, or documentation data.

#### Scenario: Test without a mirrored module
- **WHEN** the test modules under `tests` are listed
- **THEN** each one names a module of `skcomplex` at the mirrored path, and none lies outside `tests/core` and
  `tests/systems`

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

### Requirement: The package holds two subpackages
The package SHALL expose exactly two subpackages, `skcomplex.core` and `skcomplex.systems`, and no module beside them:
- `skcomplex.core.base`, `skcomplex.core.types`, `skcomplex.core.specs`, and `skcomplex.core.errors` hold the root
  class `BaseConfig`, the type aliases, the specs, and the errors.
- `skcomplex.core.components` holds what a system is: `base`, `spaces`, `laws`, `elements`, `interactions`, `system`,
  and the types and specs that name them.
- `skcomplex.core.engine` holds what turns a system into a trajectory: `problems`, `compile`, `backends`, `solvers`,
  `runtime`, `translators`, and `viz`.
- `skcomplex.core.analysis` holds what measures a trajectory, an array, or a system.
- `skcomplex.core.experiments` holds what runs many systems.
- `skcomplex.systems` holds the catalog of ready-made systems.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components.elements`, `simulate` from
  `skcomplex.core.engine.runtime`, `RK4Solver` from `skcomplex.core.engine.solvers`, `entropy` from
  `skcomplex.core.analysis`, and `sweep` from `skcomplex.core.experiments`
- **THEN** every import succeeds

#### Scenario: Old paths
- **WHEN** a user imports `skcomplex.core.engine`, `skcomplex.core.analysis`, `skcomplex.core.errors`, or `skcomplex.systems`
- **THEN** the import fails, since the old paths keep no alias

### Requirement: The core problems package holds what a system compiles to
`skcomplex.core.components.problems` SHALL hold the compiler's `Problem`, the kernels, reductions, neighbourhoods, and printing
that a problem's numeric functions are written from, the `Lowering` a backend returns, and `BaseBackend` with the
registry of backends. It SHALL import no package of `skcomplex.core.engine`, and the engine SHALL register the loader that
imports its backends.

#### Scenario: Backend base
- **WHEN** a user imports `BaseBackend`, `Lowering`, and `Problem` from `skcomplex.core.components.problems`
- **THEN** every import succeeds, and `NumbaBackend` of `skcomplex.core.engine.backends` derives from `BaseBackend`

#### Scenario: Backends load on demand
- **WHEN** a system is compiled with the backend `'numba'` and `skcomplex.core.engine.backends` was never imported
- **THEN** the registry of backends imports it through the loader the engine registered, and the compile succeeds

### Requirement: The runtime reads the base problem
The runtime SHALL run any problem through what `BaseProblem` holds, and SHALL read the compiler's fields only of a
`Problem`. The quantum and rewriting problems SHALL derive from `BaseProblem` alone, with the fields they use.

#### Scenario: Problem of a formalism
- **WHEN** a quantum system and a rewriting system are simulated and recorded
- **THEN** both runs succeed, and neither problem is a subclass of `skcomplex.core.components.problems.Problem`

### Requirement: A system compiler is not a pass
`BaseSystemCompiler` of `skcomplex.core.engine.compile` SHALL be the base of the compilers that claim a whole system of a
formalism and compile it to a problem, and SHALL be unrelated to `BasePass`, whose subclasses are steps of one
compilation.

#### Scenario: Quantum compiler
- **WHEN** a test checks `QuantumCompiler` against `BaseSystemCompiler` and `BasePass`
- **THEN** it derives from the first and not from the second

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseElement`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every law from `BaseLaw`, every problem from `BaseProblem`, every solver from `BaseSolver`, every backend
from `BaseBackend`, and every spec from `BaseSpec`, and every one of these bases
SHALL derive from `BaseConfig`. Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks every class whose name starts with `Base` in `skcomplex.core.components`, `skcomplex.core.engine`,
  `skcomplex.core.analysis`, `skcomplex.core.experiments`, and `skcomplex.core.specs` against `BaseConfig`
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
- **WHEN** a user imports `BaseSpace` and `BaseTime` from `skcomplex.core.components.space`, and `Layout` and `BaseProblem` from
  `skcomplex.core.components.problems`
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
`Solver`, or `Backend`, in the library, its systems, its tests, its documents, and the names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.components.laws`, `Element` and `GroupElement` from
  `skcomplex.core.components`, and `RK4Solver` from `skcomplex.core.engine.solvers`
- **THEN** every import succeeds

#### Scenario: System names
- **WHEN** every class that the system library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from, and none ends with `Component`

### Requirement: What every package shares lives in root modules
A module used by every package of the core SHALL live at the root of `skcomplex.core`, not in one package:
`skcomplex.core.base` SHALL define `BaseConfig`, the root of every configurable class, `skcomplex.core.errors` SHALL
define every error and warning, `skcomplex.core.specs` SHALL define the shared specs, and `skcomplex.core.types` SHALL
define every type alias built from Python, NumPy, SciPy, and Sage types, one alias per type, such as `Array` for every
float array. An alias that names a class of the library SHALL stay in the module of that class, and the catalog SHALL
keep its own aliases.

#### Scenario: Root imports
- **WHEN** a user imports `BaseConfig` from `skcomplex.core.base`, `ValidationError` from `skcomplex.core.errors`,
  `StateSpec` from `skcomplex.core.specs`, and `Array` from `skcomplex.core.types`
- **THEN** every import succeeds and none of these modules imports a package

#### Scenario: One alias per type
- **WHEN** a module of the components, the engine, analysis, or experiments needs a float array
- **THEN** it imports `Array` from `skcomplex.core.types` instead of defining its own alias

### Requirement: Library packages hold general operations
The core, the engine, analysis, and experiments SHALL hold general operations named mathematically: every public class
SHALL name what it computes or simulates, and its choices SHALL name variants of that operation. A name taken from one
application field SHALL appear only in the systems library, as a realization that subclasses the general class in that
field's package and fixes its settings.

#### Scenario: Field name in a library package
- **WHEN** a reviewer lists the public classes and the choices of the core, the engine, analysis, and experiments
- **THEN** none is named after a quantity of one application field, such as the Binder cumulant, the polar order, or
  the segregation index

#### Scenario: Field measure in the systems library
- **WHEN** a physics system reports the Binder cumulant of its magnetisation
- **THEN** it uses a realization of the physics field that subclasses the general distribution analysis and only fixes
  its settings

### Requirement: The engine surface is its pipeline
Every package of `skcomplex.core.engine` SHALL re-export only classes: the bases a developer subclasses, the classes of
the pipeline from a system to a trajectory, and the frozen values a client reads, with no public function. Every helper
of one class SHALL be a private method of that class.

#### Scenario: Engine surface listing
- **WHEN** a user lists the public names of every package of `skcomplex.core.engine`
- **THEN** every name is a class derived from a base of its package, and none is `compile_system`, `validate`, or a
  helper of a single pass, such as `values_of`, `member_of`, or `hilbert_of`

#### Scenario: A user solver
- **WHEN** a user subclasses the solver base, implements its hooks, and passes an instance to a simulation
- **THEN** the simulation runs with that solver

### Requirement: Every engine class follows the common API
Every class of the engine SHALL derive from exactly one base of its package, SHALL store only its settings in
`__init__`, SHALL validate them in `prepare`, SHALL take the one object it serves in `prepare(target)` and set the
attributes that end in `_`, and SHALL take the one input it acts on in `run(inputs)`. A base SHALL take no argument in
its main public methods that its peers do not take, and a variation between subclasses SHALL be a setting.

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
- **THEN** every one is frozen and has no public method besides `prepare` and `run`, and no method besides the hook `_check` and plain readers

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

### Requirement: The engine names no realization of a system package
The engine SHALL NOT list a compiler, a form, or a law kind other than those of its own packages. A form, its laws,
its compiler, and its solver that serve the systems of one field SHALL live in that field of the systems library.

#### Scenario: No realization in the engine
- **WHEN** a test searches the public names of `skcomplex.core.components`, `skcomplex.core.engine`, `skcomplex.core.specs`, and `skcomplex.core.types`
  for `KochElement`, `CantorElement`, `SierpinskiElement`, and `grigorchuk_group`
- **THEN** it finds none

#### Scenario: Rewriting in the core and the engine
- **WHEN** a user imports `WordElement` and `PatchElement` from `skcomplex.core.components.elements`, `ProductionLaw` and
  `SubstitutionLaw` from `skcomplex.core.components.laws`, and `RewritingCompiler` from `skcomplex.core.engine.compile`
- **THEN** every import succeeds, and none of these modules imports `skcomplex.systems`

### Requirement: The library calls its ready-made items systems
Every identifier, path, documentation page, and sentence of the library SHALL call a ready-made item of the catalog a
system, never a model, including scientific names such as "the SIR system". Identifiers of other libraries, such as
`mesa.Model`, and the titles of cited references SHALL keep their own wording.

#### Scenario: Catalog identifiers
- **WHEN** a user lists the public names of `skcomplex.systems`
- **THEN** they are `all_systems` and `import_extra`, and every ready-made system class ends in `System`

#### Scenario: Prose of the documentation
- **WHEN** a reviewer searches the README, the user guide, and the docstrings for the word model as a name of a
  ready-made system
- **THEN** it appears only in identifiers of other libraries and in the titles of cited references

### Requirement: A system module defines its system completely
Every module of a subfield package of `skcomplex.systems` that defines a system class SHALL hold the elements,
interactions, views, and helpers that only that system uses, so the module reads as a complete example. A subfield
package SHALL NOT hold a component module that serves a single system.

#### Scenario: Reading one system
- **WHEN** a reader opens the module that defines the class of `active_walkers`
- **THEN** the element, interaction, and view classes that only `active_walkers` uses are defined in that module

#### Scenario: Component module of one system
- **WHEN** a subfield package holds a module that defines no system class and is imported by the module of only one
  system
- **THEN** the layout test fails and names the module and the system that should hold it

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the parts a modeler defines and for calibration, and public functions for
simulating, analysing, and running experiments. Every other helper SHALL be a private method of the class that uses it.
Compilers and problem forms SHALL be private, with one documented hook for a system package that brings its own form.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, no concrete compiler or `Simulation`,
  and of compilation and problems only the bases a system package's own form builds on, the problem of the shared pass
  core, and the frames observers read

#### Scenario: A system package brings its own form
- **WHEN** a system class tunes the run setting `compiler` with a compiler its package defines on `BaseCompiler`
- **THEN** `simulate` compiles the system with it, while the library's own forms, such as the rewriting and operator
  forms, are chosen from what the system holds
