## MODIFIED Requirements

### Requirement: General elements
The core SHALL provide `NonnegativeElement`, which holds one scalar state `x` in `[0, oo)`, and `PointElement`, a
member with a vector state of order 2 in a continuous space. A compartment, a density at one class, a species, or a
population SHALL be `NonnegativeElement` or a field realization that subclasses it and keeps the state `x`. The core
SHALL provide no second element for a nonnegative scalar under another name or another state name.
`MassActionSystem` SHALL build one `NonnegativeElement` per species and key its reactions by `(species, 'x')`, and
`EncounterInteraction` SHALL read the state `x` unless it is given another.

#### Scenario: Kinetic systems share one element
- **WHEN** the kinetic systems of wealth, opinion, crowds, immune competition, and gases declare their densities
- **THEN** every density is a `NonnegativeElement`, through the realization of its own field, and its state is `x`

#### Scenario: Mass action species
- **WHEN** a user builds a `MassActionSystem` with the reaction `S + I -> 2 I` and simulates it
- **THEN** the trajectory holds `S.x` and `I.x`, and the reaction reads the keys `('S', 'x')` and `('I', 'x')`

#### Scenario: Former elements are gone
- **WHEN** a user imports `CompartmentElement` or `DensityElement` from `skcomplex.core.components.elements`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a removed class
