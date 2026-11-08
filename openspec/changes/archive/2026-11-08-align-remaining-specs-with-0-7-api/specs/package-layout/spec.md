## MODIFIED Requirements

### Requirement: A system compiler is not a pass
`BaseCompiler` of `skcomplex.core.engine.compile` SHALL be the public base of the compilers that compile a whole system
of one form to a problem, the private compilers of the library's forms included, and SHALL be unrelated to `BasePass`,
whose subclasses are steps of one compilation.

#### Scenario: Quantum compiler
- **WHEN** a test checks the compiler that `simulate` chooses for a quantum system against `BaseCompiler` and `BasePass`
- **THEN** it derives from the first and not from the second

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
