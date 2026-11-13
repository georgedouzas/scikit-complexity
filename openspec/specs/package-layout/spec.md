# package-layout Specification

## Purpose
Separates the domain-agnostic framework, the core, from the domain system library, so every system is built only from
the general interface and the interface never depends on a system.

## Requirements

### Requirement: Dependencies run from experiments to analysis to engine to components
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

### Requirement: The engine problems package holds what a system compiles to
`skcomplex.core.engine.problems` SHALL hold `BaseProblemSpec`, the compiler's `ProblemSpec`, and the kernels, printers,
and searches that a problem's numeric functions are written from. It SHALL import no other package of
`skcomplex.core.engine` than `types`. A backend SHALL be passed to compilation as an instance, and without one the
compiler SHALL choose `NumbaBackend` for large systems and `NumpyBackend` for the others.

#### Scenario: Problem bases
- **WHEN** a user imports `BaseProblemSpec`, `ProblemSpec`, and `BaseKernel` from `skcomplex.core.engine.problems`
- **THEN** every import succeeds

#### Scenario: Backend chosen by size
- **WHEN** a system whose groups hold 10 members in total is compiled without a backend
- **THEN** the compiled problem runs on `NumpyBackend`

### Requirement: The runtime reads the base problem
The runtime SHALL run any problem through what `BaseProblemSpec` holds, and SHALL read the compiler's fields only of
a `ProblemSpec`. The operator and rewriting problems SHALL derive from `BaseProblemSpec` alone, with the fields they
use.

#### Scenario: Problems beyond the compiler's
- **WHEN** a quantum system and a rewriting system are simulated and recorded
- **THEN** both runs succeed, and neither problem is a subclass of `skcomplex.core.engine.problems.ProblemSpec`

### Requirement: A system compiler is not a pass
`BaseCompiler` of `skcomplex.core.engine.compile` SHALL be the public base of the compilers that compile a whole system
of one form to a problem, the compilers of the library's forms included, and SHALL be unrelated to `BasePass`,
whose subclasses are steps of one compilation.

#### Scenario: Quantum compiler
- **WHEN** a test checks the compiler that `simulate` chooses for a quantum system against `BaseCompiler` and `BasePass`
- **THEN** it derives from the first and not from the second

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseElement`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every law from `BaseLaw`, every problem from `BaseProblemSpec`, every solver from `BaseSolver`, every backend
from `BaseBackend`, and every spec from `BaseSpec`, and every one of these bases
SHALL derive from `BaseConfig`. Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks every class whose name starts with `Base` in `skcomplex.core.components`, `skcomplex.core.engine`,
  `skcomplex.core.analysis`, `skcomplex.core.experiments`, and `skcomplex.core.specs` against `BaseConfig`
- **THEN** every check holds

#### Scenario: Domain of a state
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and prepared with a `BaseProblemSpec` of its form
- **THEN** it advances the problem through `run`

### Requirement: Bases are unrelated
No base class SHALL derive from another, except that `BaseElement`, `BaseInteraction`, `BaseLaw`,
`BaseNeighborhood`, `BaseSpace`, `BaseTime`, and `BaseSystem` SHALL derive from `BaseComponent`, and that every
configurable base, such as `BaseComponent`, `BaseSolver`, `BaseBackend`, and `BaseCompiler`, SHALL derive from
`BaseConfig`, which derives from scikit-learn's `BaseEstimator`. `BaseSpec` SHALL derive from no base.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another, except the bases of components, which are subclasses of
  `BaseComponent`, and the configurable bases, which are subclasses of `BaseConfig`

#### Scenario: System and interaction
- **WHEN** a test checks `CompositeSystem` and `BaseInteraction` against `BaseElement`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: Classes are named after their base
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `CompositeSystem`, and `Schedule`, SHALL
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
  `SubstitutionLaw` from `skcomplex.core.components.laws`, and `RewritingSolver` from `skcomplex.core.engine.solvers`
- **THEN** every import succeeds, none of these modules imports `skcomplex.systems`, and the public names of
  `skcomplex.core.engine.compile` hold no compiler of a form of the library

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
- **WHEN** a reader opens the module that defines the class of `ActiveWalkersSystem`
- **THEN** the element, interaction, and view classes that only `ActiveWalkersSystem` uses are defined in that module

#### Scenario: Component module of one system
- **WHEN** a subfield package holds a module that defines no system class and is imported by the module of only one
  system
- **THEN** the layout test fails and names the module and the system that should hold it

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the components a modeler defines and for calibration, and public functions
for simulating, analysing, and running experiments. Every other helper SHALL be a method of the class that reads it,
with only the module functions that the requirement on classes owning the functions they read lets stay. Compilers and
problem forms SHALL be private, with one documented hook for a system package that brings its own form, except the
operator compiler and the operator problem, which SHALL be public so that the quantum analyses of a field package can
compile an operator system and read its problem.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, no concrete compiler but
  `OperatorCompiler` and no `Simulation`, and of compilation and problems only the bases a system package's own form
  builds on, the problems of the shared pass core and of the operator form, and the frames observers read

#### Scenario: A system package brings its own form
- **WHEN** a system class tunes the run setting `compiler` with a compiler its package defines on `BaseCompiler`
- **THEN** `simulate` compiles the system with it, while the library's own forms, such as the rewriting and operator
  forms, are chosen from what the system holds

### Requirement: Systems of the core form a components package
The systems of the core SHALL live in the package `skcomplex.core.components.systems`, laid out like the packages of
the other components of a system: a `_base` module with the base, one private module per class, and an `__init__`
that re-exports the public names. The package SHALL re-export `BaseSystem`, `RunSettings`, and `CompositeSystem`.
`skcomplex.core.components` SHALL hold no module of systems outside this package.

#### Scenario: Import the systems of the core
- **WHEN** a user imports `BaseSystem`, `RunSettings`, and `CompositeSystem` from `skcomplex.core.components.systems`
- **THEN** every import succeeds, and the modules of the package are `_base` and `_composite`

#### Scenario: Former modules are gone
- **WHEN** a user imports `skcomplex.core.components.system` or `skcomplex.core.components.mass_action`
- **THEN** the import fails with `ModuleNotFoundError`, since the project ships no alias for a moved module

#### Scenario: Former class names are gone
- **WHEN** a user imports `System`, `MassActionSystem`, or `ReactionNetworkSystem` from
  `skcomplex.core.components.systems`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a renamed or moved class

#### Scenario: Systems import the public package
- **WHEN** a module of `skcomplex.systems` imports `BaseSystem`, `RunSettings`, or `CompositeSystem`
- **THEN** it imports them from `skcomplex.core.components.systems`, and the import contracts of the checks session
  pass

### Requirement: The base of systems imports no other module of its package
The `_base` module of `skcomplex.core.components.systems` SHALL define `BaseSystem` and `RunSettings` and SHALL import
no other module of its package. A concrete system module SHALL import the base downward.

#### Scenario: Base module stands alone
- **WHEN** the imports of `skcomplex.core.components.systems._base` are listed
- **THEN** none of them names `_composite`, `_reaction_network`, or any other module of the package

#### Scenario: Building a system that is not a composite
- **WHEN** a subclass of `BaseSystem` builds its components as a `BaseSystem` that is not its own composite and is
  prepared
- **THEN** `prepare` raises `ValidationError` naming the class, as before the move

#### Scenario: Composite of a ready-made system
- **WHEN** `LotkaVolterraSystem().prepare().system_` is read
- **THEN** it is an instance of `CompositeSystem` imported from `skcomplex.core.components.systems`

### Requirement: Core elements and neighbourhoods hold no parametrised copies
The elements and spaces of the core SHALL hold no class that only fixes the arguments, the state name, or the order of
another core class. Such a class SHALL become an argument of the general class, or a realization in the systems that
subclasses the general class: in the package of the field that has its own term for it, or in
`skcomplex.systems.general` when several fields share it. A pure math convenience that the core does not build itself
SHALL get no class.

#### Scenario: Nonnegative scalar in a field
- **WHEN** a biology, physics, chemistry, economics, or social system needs a compartment, a density, or a species
- **THEN** it uses `NonnegativeElement` of `skcomplex.systems.general.elements` or a realization of its own field that
  subclasses it, and no two fields share one realization

#### Scenario: Pure math convenience
- **WHEN** a system needs the Moore neighbourhood, the von Neumann neighbourhood, or a group on the sites of a lattice
- **THEN** it uses `BallNeighborhood` with a metric, which the lattice also builds as its default neighbourhood, or
  `GroupElement` on `'sites'`, and no field package defines a copy of either

#### Scenario: Cones are radius neighbourhoods
- **WHEN** a flock looks at the neighbours within a radius and an angle around its heading
- **THEN** it uses `RadiusNeighborhood(radius, angle=angle, direction='theta')`, the pairs equal those of the former
  `ConeNeighborhood`, and `RadiusNeighborhood(radius)` has the angle `pi`

### Requirement: Compile records shared across engine packages are public
A spec of `skcomplex.core.components.specs` that a module of another package imports SHALL be public and exported by
its package, and no module SHALL import a private name of another package.

#### Scenario: Subsystem spec
- **WHEN** the operator compiler and the operator lowering name the subsystem of a tensor product
- **THEN** both import the public `SubsystemSpec` from `skcomplex.core.components.specs`

### Requirement: Every component has one place by the placement rule
Every new component SHALL be placed by one ordered test. A structural base SHALL go to the core. A component the
engine handles by type, or an operator generated from one function, SHALL go to the core under a mathematical name.
Any other component is a recipe, placed by its users: `skcomplex.systems.general.<concept>` for several fields,
`systems/<field>/common/<family>` for one field, its system's module for one system. Every analysis SHALL be placed by
its users too: an analysis that measures what systems of several fields share SHALL go to `skcomplex.core.analysis`,
and an analysis of one field SHALL go to `systems/<field>/common/<family>`. The core SHALL never import
`skcomplex.systems`.

#### Scenario: Structural base
- **WHEN** a contributor adds a base where laws run once per triangle of a cell complex
- **THEN** it goes to the core, and the source tests accept it as a class that another public core class subclasses

#### Scenario: Handled by type
- **WHEN** a contributor adds a law that a new kernel lowers
- **THEN** it goes to the core under a mathematical name, and the source tests accept it because the engine names it

#### Scenario: Recipe of several fields
- **WHEN** a contributor adds an agent rule that systems of economics and social use
- **THEN** it goes to a concept subpackage of `skcomplex.systems.general`, and the source tests reject it in the core,
  since no engine code names it and it returns only `RuleLaw`

#### Scenario: Missing primitive
- **WHEN** a recipe needs an operation the engine cannot lower
- **THEN** the operation enters the core as a new primitive by the second step, and the recipe keeps no engine code

#### Scenario: Engine support keeps its place
- **WHEN** the expression printer of the engine reads the dialects of `DialectSpec`
- **THEN** `DialectSpec` stays in `core/specs/_dialect_spec.py` as engine support, and `skcomplex.core.specs` does not
  re-export it, since nothing outside the package reads it

#### Scenario: Analysis of several fields
- **WHEN** a contributor adds an analysis of the winding defects of a field of angles, which liquid crystals, active
  nematics, and spiral waves of heart tissue share
- **THEN** it goes to `skcomplex.core.analysis`

#### Scenario: Analysis of one field
- **WHEN** a contributor adds an analysis of the entanglement of the subsystems of a quantum state
- **THEN** it goes to `systems/physics/common/quantum`, and it imports only the public modules of the core

#### Scenario: Source tests guard the rule
- **WHEN** the source tests run
- **THEN** every public class of `skcomplex.core.components` is a structural base, is named in the code of another core
  module, or is the interaction with the fewest law classes among those of its base that list one of its laws other
  than `RuleLaw`, no module under `src/skcomplex/core` imports `skcomplex.systems`, and the roots of `systems/general`
  and of every `systems/<field>/common` hold no module but `__init__.py`

#### Scenario: One text of the rule
- **WHEN** a reviewer reads `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide
- **THEN** each states the same ordered test as this requirement, in the word component, and each says that analyses
  follow it by their users

### Requirement: Module constants lead the module under public names
Every module constant of a Python file of the sources, the tests, and the documents, without the generated pages,
SHALL have a public `UPPER_CASE` name and SHALL sit in one block right after the imports, before the type aliases and
the first definition. The block SHALL hold no blank line and no comment, and no constant SHALL carry an attribute
docstring. A value built from a definition of its own module SHALL follow the same rules right after that definition.

#### Scenario: Constant after a class
- **WHEN** a module defines a literal constant below its first class or function
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Private constant
- **WHEN** a module defines a constant whose name begins with an underscore
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Gap in the block
- **WHEN** a blank line, a comment, or a string sits between two constants of a block, or a comment sits right above it
- **THEN** the source test fails and names the file and the line

#### Scenario: Test and example files
- **WHEN** a test module or a gallery example defines a constant after its first function
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Derived value
- **WHEN** a module builds a dispatch table from functions defined in the same module
- **THEN** the table sits right after the last of them under a public name, and the source test passes

#### Scenario: Renamed constants stay off the surface
- **WHEN** a test lists the names every package re-exports, before and after the renames
- **THEN** the two lists are equal

### Requirement: Classes own the functions they read
A module function of the sources that a class reads, directly or through other module functions, in its module or in
a module that imports it, SHALL be a method of one owning class, which the other readers call through. Only
Numba-compiled kernels, helpers that only public functions read, and functions a derived value reads SHALL stay module
functions. A public function SHALL be re-exported by its package, live in a `_utils` module, or be compiled.

#### Scenario: Helper of one class
- **WHEN** a private module function is read only by one class of its module
- **THEN** the source test fails and names the function and the class that should own it

#### Scenario: Helper of several classes
- **WHEN** two system classes of one module read the same private module function
- **THEN** the source test fails, and after the fix one class owns it as a static method and the other calls it through
  the owner

#### Scenario: Helper imported by a class of another module
- **WHEN** a pass class imports a private function of another module of its package
- **THEN** the source test fails and names the function, the importing module, and the reading class

#### Scenario: Numba kernel
- **WHEN** a module defines Numba-compiled functions that call one another
- **THEN** they stay module functions and the source test passes

#### Scenario: Helper of an analysis function
- **WHEN** a private function of an analysis module is read only by the public analysis function and its private helpers
- **THEN** it stays a private module function and the source test passes

#### Scenario: Unexported public helper
- **WHEN** a private module defines a public function that no package re-exports and that is not compiled and not in a
  `_utils` module
- **THEN** the source test fails and names the function

#### Scenario: Behaviour after the move
- **WHEN** the test suite runs after the helpers move into classes
- **THEN** every test that passed before passes, with only imports of moved private names changed

### Requirement: Methods begin with a verb
Every method of the sources SHALL begin with a verb, as the constitution asks, apart from magic methods, the hooks a
library calls by a fixed name, properties, and names a base class of another library defines. A helper moved into a
class whose name begins with a noun SHALL take a verb name.

#### Scenario: Moved helper with a noun name
- **WHEN** the private function `_noise` of a reaction-diffusion module becomes a method of its system class
- **THEN** it is named `_get_noise`, and every call site uses the new name

#### Scenario: Method that begins with a noun
- **WHEN** a class of the sources defines a method whose first word is not a known verb
- **THEN** the source test fails and names the file, the line, and the method

#### Scenario: Library hook
- **WHEN** a symbolic function class defines `_eval_` or `_print_`
- **THEN** the source test passes

### Requirement: Symbolic functions are BuiltinFunction subclasses
Every symbolic function the library defines SHALL be an instance of a private `BuiltinFunction` subclass whose
printing, evaluation, and derivative are methods. Sage's `function` factory SHALL be used only to declare an unknown
function of an equation by its name. Expressions SHALL print and compile as before.

#### Scenario: Function with a print hook
- **WHEN** a reviewer reads how the neighbour reduction prints
- **THEN** it is the `_print_` method of its subclass, and no module function holds it

#### Scenario: Factory call with a hook
- **WHEN** a module of the sources calls Sage's `function` at module level or with a hook keyword
- **THEN** the source test fails and names the file and the line

#### Scenario: Unknown function
- **WHEN** a cosmology declares the scale factor with `function('a')` inside a method
- **THEN** the source test passes and the metric reads it as an unknown

#### Scenario: Printing unchanged
- **WHEN** the printer and compile tests run after the conversion
- **THEN** every expression prints, differentiates, and compiles as before

### Requirement: The components package holds the base of every component
`skcomplex.core.components.base` SHALL define `BaseComponent`, the base of every component, and SHALL import only the
root modules of `skcomplex.core`. The base of each family of components SHALL live in the `_base` module of its
package: `BaseElement` in `elements`, `BaseInteraction` in `interactions`, `BaseLaw` in `laws`, `BaseNeighborhood`,
`BaseSpace`, and `BaseTime` in `spaces`, and `BaseSystem` in `systems`. The base of any other concept SHALL live in
that concept's package.

#### Scenario: The bases of components
- **WHEN** a user imports `BaseComponent` from `skcomplex.core.components.base`, and `BaseElement`,
  `BaseInteraction`, `BaseLaw`, `BaseSpace`, `BaseTime`, and `BaseSystem` from their packages
- **THEN** every import succeeds, and every base but `BaseComponent` is a subclass of `BaseComponent`

#### Scenario: Other bases
- **WHEN** a user imports `BaseSolver` from `skcomplex.core.engine.solvers` and `BaseBackend` from
  `skcomplex.core.engine.backends`
- **THEN** every import succeeds, and neither is a subclass of `BaseComponent`

### Requirement: One word names the building blocks of a system
The building blocks of a system, which are its elements, interactions, laws, spaces, neighbourhoods, times, and the
system itself, SHALL be called components in names, messages, docstrings, documentation pages, specs, and plans.
Part SHALL keep only its other meanings. A mathematical component, such as a vector, tensor, or connected component,
SHALL be qualified in prose, so that the bare word names a building block.

#### Scenario: The base of every component
- **WHEN** a user imports `BaseComponent` from `skcomplex.core.components.base`
- **THEN** the import succeeds, and `BaseElement`, `BaseInteraction`, `BaseLaw`, `BaseNeighborhood`, `BaseSpace`,
  `BaseTime`, and `BaseSystem` are subclasses of it

#### Scenario: Other meanings of part
- **WHEN** a reader searches the library for part
- **THEN** every match is a real or imaginary part, a part of a tensor or of the rates, a site, edge, or hyperedge of
  a space, a piece of a delay, a word, a patch, or a string, a domain word of one model, or ordinary English

### Requirement: The convention check keeps one word
`openspec/scripts/check_conventions.py` SHALL flag part used for a building block of a system in the names of `src`
and `tests`, in the docstrings and comments of `src`, and in the documentation pages, with an allow list of the kept
meanings, and SHALL hold a self-check against planted files.

#### Scenario: A planted part
- **WHEN** the convention check runs on a planted module that defines `DoublingPart(BaseComponent)` with the docstring
  `Return the parts of the system.` and calls `real_part`
- **THEN** it names the class and the docstring, and does not name `real_part`

#### Scenario: Kept meanings pass
- **WHEN** the convention check runs on the repository
- **THEN** `real_part`, `symmetric_part`, `PARTS` of a group, `delay_parts`, `_count_parts`, and the parts of the
  watchmakers raise no finding
