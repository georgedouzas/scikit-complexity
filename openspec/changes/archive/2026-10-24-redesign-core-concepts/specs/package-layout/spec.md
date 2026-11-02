# Spec Delta

## MODIFIED Requirements

### Requirement: The core holds every domain-agnostic package
The package SHALL expose exactly two subpackages: `skcomplex.core`, holding every domain-agnostic package, and
`skcomplex.models`, holding the domain models. The core's packages SHALL be `base`, `space`, `laws`, `components`,
`interactions`, `system`, `compile`, `solvers`, `backends`, `runtime`, `viz`, `experiments`, `analysis`, and
`formalisms`. Each package SHALL import only the packages listed before it.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components`, `Simulation` from `skcomplex.core.runtime`, `RK4`
  from `skcomplex.core.solvers`, and `Domain` from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old framework paths are gone
- **WHEN** a user imports `skcomplex.model`, `skcomplex.core.model`, `skcomplex.solvers`, or `skcomplex.runtime`
- **THEN** the import fails

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.core.solvers` imports from `skcomplex.core.backends`
- **THEN** the layout check fails and names the module and the package it reached

## ADDED Requirements

### Requirement: Modules hold one concept in a fixed order
Every module of `skcomplex.core` SHALL hold one concept, SHALL define private functions, public functions, private
classes, and public classes in that order unless a name is needed before its turn, and SHALL NOT define a helper that
serves a concept owned by another module. The module that defines a base class of components or laws SHALL import only
`skcomplex.core.base`. No module SHALL guard an import behind a type-checking condition.

#### Scenario: Helper between classes
- **WHEN** a module defines a private function after a public class
- **THEN** the layout check fails and names the module and the function

#### Scenario: Base class with wide imports
- **WHEN** the module that defines `Component` imports from `skcomplex.core.space`
- **THEN** the layout check fails and names the module

#### Scenario: Guarded import
- **WHEN** a core module imports a name under `if TYPE_CHECKING:`
- **THEN** the layout check fails and names the module

### Requirement: Packages export only what clients use
A core package SHALL re-export only the names that the models, the tests, or the documentation use, or that another
core package uses as a concept of the exporting package. A helper that only one other package uses SHALL move into that
package, so no package exports the internals of another.

#### Scenario: Compiler internal leaks
- **WHEN** `skcomplex.core.laws` lists in `__all__` a helper that only `skcomplex.core.compile` uses
- **THEN** the export check fails and names the package and the helper

### Requirement: The test tree mirrors the source tree
Every test module other than the repository-wide checks SHALL test one source module and SHALL live at the path that
mirrors it, so `tests/core/<package>/test_<module>.py` tests `skcomplex.core.<package>._<module>`, and
`tests/models/<field>/<subfield>/test_<module>.py` tests the module of that model package.

#### Scenario: Test without a mirrored module
- **WHEN** `tests/core/compile/test_compiler.py` exists and `skcomplex.core.compile` has no module `_compiler`
- **THEN** the layout check fails and names the test module
