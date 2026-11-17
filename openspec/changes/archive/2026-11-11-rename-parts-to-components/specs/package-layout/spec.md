## ADDED Requirements

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

## RENAMED Requirements

- FROM: `### Requirement: Every part has one place by the placement rule`
- TO: `### Requirement: Every component has one place by the placement rule`

## REMOVED Requirements

### Requirement: The core base holds the bases of a system's parts
**Reason**: It described a package `skcomplex.core.base` with a module `_parts` that no longer exists, and called the building blocks of a system parts.
**Migration**: See the requirement "The components package holds the base of every component", which replaces it.

## MODIFIED Requirements

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

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the components a modeler defines and for calibration, and public functions
for simulating, analysing, and running experiments. Every other helper SHALL be a method of the class that reads it,
with only the module functions that the requirement on classes owning the functions they read lets stay. Compilers and
problem forms SHALL be private, with one documented hook for a system package that brings its own form.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, no concrete compiler or `Simulation`,
  and of compilation and problems only the bases a system package's own form builds on, the problem of the shared pass
  core, and the frames observers read

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

### Requirement: Every component has one place by the placement rule
Every new component SHALL be placed by one ordered test. A structural base SHALL go to the core. A component the
engine handles by type, or an operator generated from one function, SHALL go to the core under a mathematical name.
Any other component is a recipe, placed by its users: `skcomplex.systems.general.<concept>` for several fields,
`systems/<field>/common/<family>` for one field, its system's module for one system. The core SHALL never import
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

#### Scenario: Source tests guard the rule
- **WHEN** the source tests run
- **THEN** every public class of `skcomplex.core.components` is a structural base, is named in the code of another core
  module, or is the interaction with the fewest law classes among those of its base that list one of its laws other
  than `RuleLaw`, no module under `src/skcomplex/core` imports `skcomplex.systems`, and the roots of `systems/general`
  and of every `systems/<field>/common` hold no module but `__init__.py`

#### Scenario: One text of the rule
- **WHEN** a reviewer reads `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide
- **THEN** each states the same ordered test as this requirement, in the word component
