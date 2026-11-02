## ADDED Requirements

### Requirement: Every element derives from one base
Elements and groups SHALL derive from `BaseElement`, which holds a label, declares states, parameters, and roles, and
returns laws. A modeller SHALL define a type of element by subclassing `Element`.

#### Scenario: Hierarchy
- **WHEN** a user checks `Element`, `GroupElement`, and `FieldElement` against `BaseElement`
- **THEN** each is a subclass of `BaseElement`, and `System` and `BaseInteraction` are not

### Requirement: A system is defined by elements, interactions, and laws
A system SHALL be defined by its elements, its interactions, and the laws they return, and SHALL declare no other
kind of part. A value that only names an expression SHALL be written in the laws, and what a run records or checks
SHALL be an argument of the run.

#### Scenario: Auxiliary as an expression
- **WHEN** a law computes `leverage = s.debt / s.equity` and uses it in a rate, and the run records an observable of
  the same expression
- **THEN** the trajectory holds `leverage`, and the system has no quantity among its parts

### Requirement: Runs check invariants
`Simulation(invariants={name: tolerance})` SHALL check, at every frame, that each named observable stays within its
tolerance of its first value, and SHALL raise `ConservationError` naming the observable and the first frame that
breaks it.

#### Scenario: Violated invariant
- **WHEN** a law creates wealth from nothing in a system whose run declares the observable `total_wealth` invariant
- **THEN** the run raises `ConservationError` naming `total_wealth` and the first frame that breaks it

### Requirement: Parts share three verbs
Elements, interactions, laws, and systems SHALL derive from `BasePart`, which defines `label`, `check()`,
`prepare(namespace)`, and `run(input)`. Elements, interactions, and systems SHALL return `self` from `check` and
`prepare`, and their laws from `run(s)`; a system SHALL return the laws of its children. A law SHALL return a checked
or prepared copy, and SHALL evaluate its contribution from `run(values)`.

#### Scenario: System yields laws
- **WHEN** a system of a population element and a growth interaction is prepared and run with a namespace
- **THEN** it returns the laws of the element and of the interaction, in order

#### Scenario: Law run alone
- **WHEN** a user prepares `RateLaw(s.n, r * s.n)` against a namespace with `n` and `r`, and runs it at `n=10`,
  `r=0.5`
- **THEN** the result is `5.0`, its `targets_` holds `n`, and its `reads_` holds `n` and `r`

## REMOVED Requirements

### Requirement: Quantities are declared apart from laws
**Reason**: Quantities are not part of a system's definition, so `BaseQuantity` and `quantities(s)` are removed.
**Migration**: Write auxiliaries as expressions in `laws`, pass observables and invariants to `Simulation`, pass
objectives to the optimal control solvers, and read aggregates as reductions.

### Requirement: Invariants declare conserved quantities
**Reason**: An invariant is a check of a run, not a part of a system.
**Migration**: Pass `Simulation(observables=[...], invariants={name: tolerance})`.

### Requirement: Every component derives from one base
**Reason**: Replaced by "Every element derives from one base", with `BaseElement`.
**Migration**: Subclass `Element` for a type of element.
