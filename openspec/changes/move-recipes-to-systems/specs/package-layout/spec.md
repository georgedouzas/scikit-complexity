## MODIFIED Requirements

### Requirement: Systems of the core form a components package
The systems of the core SHALL live in the package `skcomplex.core.components.systems`, laid out like the other parts
of a system: a `_base` module with the base, one private module per class, and an `__init__` that re-exports the
public names. The package SHALL re-export `BaseSystem`, `RunSettings`, and `CompositeSystem`.
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

## ADDED Requirements

### Requirement: Every part has one place by the placement rule
Every new part SHALL be placed by one ordered test. A structural base SHALL go to the core. A part the engine handles by
type, or an operator generated from one function, SHALL go to the core under a mathematical name. Any other part is a
recipe, placed by its users: `skcomplex.systems.general.<concept>` for several fields,
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

#### Scenario: Source tests guard the rule
- **WHEN** the source tests run
- **THEN** every public class of `skcomplex.core.components` is a structural base, is named in the code of another core
  module, or is the interaction with the fewest law classes among those of its base that list one of its laws other
  than `RuleLaw`, no module under `src/skcomplex/core` imports `skcomplex.systems`, and the roots of `systems/general`
  and of every `systems/<field>/common` hold no module but `__init__.py`

#### Scenario: One text of the rule
- **WHEN** a reviewer reads `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of the user guide
- **THEN** each states the same ordered test as this requirement
