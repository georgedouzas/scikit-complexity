## ADDED Requirements

### Requirement: Core elements and neighbourhoods hold no parametrised copies
The elements and spaces of the core SHALL hold no class that only fixes the arguments, the state name, or the order of
another core class. Such a class SHALL become an argument of the general class, or, when a field has its own term for
it, a realization in that field's package that subclasses the general class. A pure math convenience SHALL get no
realization, and systems SHALL use the general class directly.

#### Scenario: Nonnegative scalar in a field
- **WHEN** a biology, physics, chemistry, economics, or social system needs a compartment, a density, or a species
- **THEN** it uses a realization of its own field that subclasses `NonnegativeElement`, and no two fields share one

#### Scenario: Pure math convenience
- **WHEN** a system needs the Moore neighbourhood, the von Neumann neighbourhood, or a group on the sites of a lattice
- **THEN** it uses `BallNeighborhood` with a metric, or `GroupElement` on `'sites'`, and no field package defines a
  copy of either

### Requirement: Compile records shared across engine packages are public
A spec of `skcomplex.core.components.specs` that a module of another package imports SHALL be public and exported by
its package, and no module SHALL import a private name of another package.

#### Scenario: Subsystem spec
- **WHEN** the operator compiler and the operator lowering name the subsystem of a tensor product
- **THEN** both import the public `SubsystemSpec` from `skcomplex.core.components.specs`
